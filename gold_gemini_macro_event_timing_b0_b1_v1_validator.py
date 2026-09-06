"""Adversarial validator for the frozen B0/B1 macro information test."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

import gold_gemini_execution_semantics_v1 as semantics
import gold_gemini_macro_event_timing_b0_b1_v1 as experiment
import training_run_history as archive

ROOT = Path(__file__).resolve().parent
CHECKS = ("chronology", "feature leakage", "label maturity", "OOF predictions", "calibration",
          "threshold selection", "purge/embargo", "holdout contamination", "recent-period reuse",
          "execution alignment", "cost assumptions", "multiple-testing risk")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def array_hash(values: np.ndarray) -> str:
    a = np.ascontiguousarray(values)
    return hashlib.sha256(str(a.dtype).encode()+np.asarray(a.shape,dtype=np.int64).tobytes()+a.tobytes()).hexdigest()


def close(a: Any, b: Any, tolerance: float = 1e-7) -> bool:
    if a is None or b is None: return a is None and b is None
    if math.isinf(float(a)) or math.isinf(float(b)): return math.isinf(float(a)) and math.isinf(float(b))
    return math.isclose(float(a),float(b),rel_tol=tolerance,abs_tol=tolerance)


def reward(values: np.ndarray) -> dict[str, Any]:
    values=np.asarray(values,dtype=float); winners=values[values>0]; losers=values[values<=0]
    gains=float(winners.sum()); loss=-float(losers.sum()); avgw=float(winners.mean()) if len(winners) else None; avgl=float(losers.mean()) if len(losers) else None
    payoff=avgw/abs(avgl) if avgw is not None and avgl not in (None,0) else None
    equity=np.r_[0,np.cumsum(values)]; dd=equity-np.maximum.accumulate(equity)
    wr=len(winners)/len(values) if len(values) else 0
    return {"trades":len(values),"wins":len(winners),"losses":len(losers),"realized_wr":wr,
        "average_winner_r":avgw,"average_loser_r":avgl,"payoff_ratio":payoff,
        "break_even_wr":1/(1+payoff) if payoff is not None else None,
        "break_even_adjusted_edge":wr-1/(1+payoff) if payoff is not None else None,
        "pf":math.inf if loss==0 and gains>0 else (gains/loss if loss else 0),"mean_r":float(values.mean()) if len(values) else 0,
        "pnl_r":float(values.sum()),"max_dd_r":float(dd.min())}


def reconstruct_foundation() -> tuple[np.ndarray,np.ndarray,list[str]]:
    foundation=experiment.FOUNDATION
    metrics=experiment.base.read_json(foundation/"metrics.json")
    errors=[]
    if archive.validate_run(foundation): errors.append("foundation FINALIZED provenance failure")
    if experiment.base.sha256(foundation/"canonical_macro_events.csv")!=experiment.EXPECTED_MACRO_DATASET_SHA: errors.append("macro dataset bytes mismatch")
    events=pd.read_csv(foundation/"canonical_macro_events.csv")
    events["release_ns"]=pd.to_datetime(events.release_timestamp_utc,utc=True).dt.as_unit("ns").astype("int64")
    events=events.sort_values(["release_ns","event_type"]).drop_duplicates("release_ns",keep="last")
    with np.load(foundation/"event_features.npz",allow_pickle=False) as saved:
        times=saved["utc_ns"].astype(np.int64); matrix=saved["features"].astype(float); names=saved["feature_names"].astype(str).tolist(); incomplete=saved["incomplete_mask"]
    ix=np.searchsorted(events.release_ns.to_numpy(),times,side="right")-1
    valid=ix>=0; age=np.full(len(times),np.inf); release=events.release_ns.to_numpy(); types=events.event_type.to_numpy()
    age[valid]=(times[valid]-release[ix[valid]])/60_000_000_000
    rebuilt=np.zeros((len(times),8)); rebuilt[:,0]=np.minimum(age,1440)/1440
    rebuilt[:,1]=age<15; rebuilt[:,2]=(age>=15)&(age<60); rebuilt[:,3]=(age>=60)&(age<240)
    for j,family in enumerate(("CPI","EMPLOYMENT","PCE","FOMC")):
        rebuilt[:,4+j]=valid&(age<240)&(types[np.maximum(ix,0)]==family)
    if names!=experiment.MACRO_FEATURES or incomplete.any(): errors.append("foundation schema/incomplete mask mismatch")
    if not np.array_equal(rebuilt,matrix) or array_hash(matrix)!=experiment.EXPECTED_EVENT_MATRIX_SHA: errors.append("independent full event matrix mismatch")
    if not np.all(age[valid]>=0): errors.append("pre-release event leakage")
    return times,matrix,errors


def foundation_broker_map(matrix_times: np.ndarray,matrix: np.ndarray) -> tuple[np.ndarray,np.ndarray,list[str]]:
    broker_parts=[]; utc_parts=[]; errors=[]
    metrics=experiment.base.read_json(experiment.FOUNDATION/"metrics.json")
    with np.load(experiment.FOUNDATION/"exact_timestamps.npz",allow_pickle=False) as exact:
        for row in metrics["blocks"]:
            b=exact[row["block"]+"_broker_ns"].astype(np.int64); u=exact[row["block"]+"_utc_ns"].astype(np.int64)
            if len(b)!=row["rows"] or array_hash(b)!=row["timestamp_sha256"]: errors.append(row["block"]+" timestamp mismatch")
            broker_parts.append(b);utc_parts.append(u)
    b=np.concatenate(broker_parts);u=np.concatenate(utc_parts); order=np.argsort(b,kind="stable");b,u=b[order],u[order]
    unique,first,counts=np.unique(b,return_index=True,return_counts=True)
    for pos,count in zip(first[counts>1],counts[counts>1]):
        if not np.all(u[pos:pos+count]==u[pos]):errors.append("overlapping block UTC mismatch")
    u=u[first]; ix=np.searchsorted(matrix_times,u)
    if np.any(ix==len(matrix_times)) or not np.array_equal(matrix_times[ix],u):errors.append("broker-to-foundation alignment mismatch")
    # The experiment feeds XGBoost float32 matrices and hashes that exact input.
    return unique,matrix[ix].astype(np.float32),errors


def load_oof(run: Path,model_id: str) -> pd.DataFrame:
    with np.load(run/"paired_oof_predictions.npz",allow_pickle=False) as d:
        frame=pd.DataFrame({"TIME_DT":pd.to_datetime(d["time_ns"]),"OPEN":d["open"],"HIGH":d["high"],"LOW":d["low"],
            "CLOSE":d["close"],"ATR":d["atr"],"M1_RSI":d["rsi"],"SPREAD":d["spread"],
            "buy_prob":d["score_c0" if model_id==experiment.MODEL_IDS[0] else "score_c1"],"sell_prob":np.zeros(len(d["time_ns"]))})
    return semantics.finalize_cohort(frame,"VALIDATOR_"+model_id,offset_hours=0)


def ledgers(run: Path) -> tuple[dict[str,list[dict[str,Any]]],list[str]]:
    submitted=pd.read_csv(run/"trade_ledger.csv",low_memory=False); result={};errors=[]
    for model_id in experiment.MODEL_IDS:
        trades,_=semantics.simulate(load_oof(run,model_id),semantics.SIMULATORS[-1])
        for trade in trades:
            entry=pd.Timestamp(trade["entry_time_api"]);trade["fold"]=next(n for n,s,e in experiment.base.FOLDS if s<=entry<e)
        expected=submitted[submitted.model_id==model_id].reset_index(drop=True)
        if len(expected)!=len(trades): errors.append(model_id+" trade count mismatch")
        else:
            for i,(row,trade) in enumerate(zip(expected.to_dict("records"),trades)):
                if pd.Timestamp(row["entry_time_api"])!=pd.Timestamp(trade["entry_time_api"]) or pd.Timestamp(row["exit_time_api"])!=pd.Timestamp(trade["exit_time_api"]) or row["exit_reason"]!=trade["exit_reason"] or not close(row["net_r"],trade["net_r"]):
                    errors.append(f"{model_id} trade mismatch {i}");break
        result[model_id]=trades
    return result,errors


def validate_metrics(run: Path,all_trades: dict[str,list[dict[str,Any]]]) -> list[str]:
    submitted=pd.read_csv(run/"fold_metrics.csv");errors=[]
    days=sum(experiment.base.core.fold_days(s,e) for _,s,e in experiment.base.FOLDS)
    for model_id,trades in all_trades.items():
        scopes=[(n,[t for t in trades if t["fold"]==n],experiment.base.core.fold_days(s,e)) for n,s,e in experiment.base.FOLDS]+[("pooled",trades,days)]
        for fold,selected,n_days in scopes:
            stats=reward(np.array([t["net_r"] for t in selected])); stress=reward(np.array([t["stress_r"] for t in selected]))
            stats.update(trades_per_day=len(selected)/n_days,tp_first_wr=sum(t["exit_reason"]=="take_profit" for t in selected)/len(selected) if selected else 0,cost_stress_pf=stress["pf"])
            row=submitted[(submitted.model_id==model_id)&(submitted.fold==fold)]
            if len(row)!=1:errors.append(f"missing metric {model_id}/{fold}");continue
            for key in ("trades","trades_per_day","realized_wr","tp_first_wr","pf","mean_r","pnl_r","max_dd_r","cost_stress_pf"):
                if not close(row.iloc[0][key],stats[key]):errors.append(f"metric {model_id}/{fold}/{key}")
    return errors


def validate_ranking(run: Path) -> list[str]:
    submitted=pd.read_csv(run/"ranking_diagnostics.csv");errors=[]
    with np.load(run/"paired_oof_predictions.npz",allow_pickle=False) as d:
        folds=d["fold_code"].astype(int);target=d["c1_target"].astype(int);net=d["c1_net_r"].astype(float)
        for model_index,model_id in enumerate(experiment.MODEL_IDS):
            score=d["score_c0" if model_index==0 else "score_c1"].astype(float)
            for code,(fold,_,_) in enumerate(experiment.base.FOLDS):
                mask=folds==code; s,y=score[mask],net[mask]; ranks=pd.Series(s).rank().to_numpy();yr=pd.Series(y).rank().to_numpy();rho=float(np.corrcoef(ranks,yr)[0,1])
                order=np.argsort(s,kind="stable"); dec=np.empty(len(s),dtype=int);dec[order]=np.minimum(np.arange(len(s))*10//len(s)+1,10)
                ten,twenty=reward(y[dec==10]),reward(y[dec>=9]);row=submitted[(submitted.model_id==model_id)&(submitted.fold==fold)].iloc[0]
                for key,value in {"spearman_score_realized_net_r":rho,"top_decile_pf":ten["pf"],"top_decile_mean_r":ten["mean_r"],"top_quintile_pf":twenty["pf"],"top_quintile_mean_r":twenty["mean_r"]}.items():
                    if not close(row[key],value,2e-6):errors.append(f"ranking {model_id}/{fold}/{key}")
    return errors


def validate(run: Path) -> dict[str,Any]:
    manifest=experiment.base.read_json(run/"manifest.json"); metrics=experiment.base.read_json(run/"metrics.json")
    provenance=experiment.base.read_json(run/"fold_model_provenance.json");errors=[]
    matrix_times,matrix,foundation_errors=reconstruct_foundation();broker,macro,map_errors=foundation_broker_map(matrix_times,matrix)
    errors+=foundation_errors+map_errors
    foundation_unchanged=experiment.inventory(experiment.FOUNDATION)==manifest["protected_finalized_runs_before"][experiment.FOUNDATION.name]
    if not foundation_unchanged:errors.append("foundation modified")
    reference={r["fold"]:r for r in experiment.base.read_json(experiment.REFERENCE/"fold_model_provenance.json")["folds"]}
    paired=True; chronology=True; hashes=True
    for fold in provenance["folds"]:
        ref=reference[fold["fold"]]
        paired &= fold["y_train_sha256_B0"]==fold["y_train_sha256_B1"]==ref["c1_label_sha256"]
        paired &= fold["x_train_sha256_B0"]==ref["x_train_sha256_C1"] and fold["x_score_sha256_B0"]==ref["x_score_sha256_C1"]
        paired &= fold["parameters_B0"]==fold["parameters_B1"]==experiment.base.FIXED_XGB_PARAMETERS and fold["random_seed_B0"]==fold["random_seed_B1"]==experiment.base.RANDOM_STATE
        chronology &= bool(fold["strict_label_maturity_before_score"]) and pd.Timestamp(fold["latest_training_label_information_time"])<pd.Timestamp(fold["score_start"])
        hashes &= fold["train_timestamp_sha256"]==ref["train_timestamp_sha256"] and fold["score_timestamp_sha256"]==ref["score_timestamp_sha256"]
        with np.load(experiment.FOUNDATION/"exact_timestamps.npz",allow_pickle=False) as exact:
            key="fold"+str([n for n,_,_ in experiment.base.FOLDS].index(fold["fold"])+1)
            hashes &= array_hash(exact[key+"_train_broker_ns"])==fold["train_timestamp_sha256"] and array_hash(exact[key+"_score_broker_ns"])==fold["score_timestamp_sha256"]
            ti=np.searchsorted(broker,exact[key+"_train_broker_ns"]);si=np.searchsorted(broker,exact[key+"_score_broker_ns"])
            hashes &= array_hash(macro[ti])==fold["macro_train_sha256"] and array_hash(macro[si])==fold["macro_score_sha256"]
    if not paired:errors.append("B0/B1 paired identity mismatch")
    if not chronology:errors.append("chronology/maturity mismatch")
    if not hashes:errors.append("timestamp/macro block hash mismatch")
    model_ok=len(provenance["models"])==6
    for item in provenance["models"]:
        path=run/item["path"];model=xgb.XGBClassifier();model.load_model(path)
        expected=provenance["b0_features" if item["model_id"]==experiment.MODEL_IDS[0] else "b1_features"]
        model_ok &= experiment.base.sha256(path)==item["sha256"] and model.get_booster().feature_names==expected
        params=model.get_params();model_ok &= params["n_estimators"]==220 and params["random_state"]==42
    if not model_ok:errors.append("model count/hash/schema/parameters mismatch")
    with np.load(run/"paired_oof_predictions.npz",allow_pickle=False) as oof:
        oof_ok=np.all(oof["feature_time_ns"]<oof["time_ns"]) and np.array_equal(oof["c1_target"],(oof["c1_net_r"]>0).astype(np.int8)) and np.isfinite(oof["score_c0"]).all() and np.isfinite(oof["score_c1"]).all()
    if not oof_ok:errors.append("OOF timing/target/score mismatch")
    all_trades,trade_errors=ledgers(run); metric_errors=validate_metrics(run,all_trades);rank_errors=validate_ranking(run)
    errors+=trade_errors+metric_errors+rank_errors
    ledger=pd.read_csv(run/"trade_ledger.csv",low_memory=False)
    denominator=ledger.sl_distance+ledger.spread_points*experiment.base.POINT
    cost_ok=np.allclose((ledger.gross_price-(ledger.spread_points+ledger.extra_cost_points)*experiment.base.POINT)/denominator,ledger.net_r,rtol=1e-9,atol=1e-9)
    if not cost_ok:errors.append("nominal cost mismatch")
    reproduction=experiment.base.read_json(run/"b0_reproduction.json");repro_ok=reproduction["pass"] and not reproduction["material_failures"]
    if not repro_ok:errors.append("B0 reproduction gate failed")
    identity=metrics["marginal_trade_identity"];set0={t["entry_time_api"] for t in all_trades[experiment.MODEL_IDS[0]]};set1={t["entry_time_api"] for t in all_trades[experiment.MODEL_IDS[1]]}
    identity_ok=identity["common_trades"]==len(set0&set1) and identity["c0_only_exact_identity"]==len(set0-set1) and identity["c1_only_exact_identity"]==len(set1-set0)
    if not identity_ok:errors.append("marginal exact trade identity mismatch")
    operational=manifest["operational_hashes_before"]==manifest["operational_hashes_after"]=={p.name:experiment.base.sha256(p) for p in (experiment.base.GEMINI_FILE,experiment.base.OPERATIONAL_MODEL)}
    if not operational:errors.append("operational artifact changed")
    no_search=manifest["search"]["performed"] is False and len(read_csv(run/"candidates.csv"))==2 and manifest["paired_design"]["threshold"]==.75 and not manifest["paired_design"]["event_subset_search"]
    if not no_search:errors.append("unauthorized search or threshold change")
    source=(run/manifest["training_script_snapshot"]).read_text(encoding="utf-8")
    code_identity=experiment.base.sha256(run/manifest["training_script_snapshot"])==manifest["training_script_sha256"] and "np.column_stack((x0_train, train_macro))" in source
    if not code_identity:errors.append("immutable execution code/only-X difference not established")
    def item(ok,evidence,reason,correction):return {"verdict":"PASS" if ok else "FAIL","evidence":evidence,"reason":reason if ok else "failed: "+reason,"required_validation_correction":"none" if ok else correction}
    internal_no_multiple=not errors
    checks={
      "chronology":item(chronology and oof_ok,"fold provenance + OOF feature times","18-month training and mature labels precede every score fold","new validation-only run with corrected time boundaries"),
      "feature leakage":item(not foundation_errors and hashes and code_identity,"certified matrix full reconstruction + six macro hashes","release<=decision and B1 is exactly B0 plus frozen matrix","reconstruct only the causal feature matrices"),
      "label maturity":item(chronology and oof_ok,"y hashes + exact maturity endpoints","same standalone S5 net-R target and strict maturity purge","correct target/maturity then new run"),
      "OOF predictions":item(oof_ok and model_ok,"six fold models + OOF arrays","each fold has distinct preceding-history B0/B1 models","regenerate genuine fold-held-out predictions"),
      "calibration":item(manifest["model"]["calibration_method"]=="none","manifest.model.calibration_method","no calibration was fitted","remove unregistered calibration"),
      "threshold selection":item(no_search,"two candidate rows + fixed manifest","0.75 and whole eight-feature family fixed; no subset/window/parameter search","discard adaptive result"),
      "purge/embargo":item(chronology,"latest label information per fold","later legacy/S5 maturity is strictly before score start","correct purge only"),
      "holdout contamination":item(not manifest["evidence_status"]["untouched_oos_claim"],"manifest evidence classification","all folds explicitly remain development evidence","remove untouched claim"),
      "recent-period reuse":item(manifest["evidence_status"]["previous_forward_status"]==experiment.base.PREVIOUS_FORWARD_STATUS,"manifest evidence status","inspected former forward data is not a fresh test","exclude prior recent evidence"),
      "execution alignment":item(not trade_errors and not metric_errors and identity_ok,"independent S5 ledgers/metrics/exact entry sets","same S5 state machine and reconciled marginal trades","correct execution reconstruction only"),
      "cost assumptions":item(cost_ok,"trade ledger arithmetic + stress metrics","observed/fallback spread and fixed nominal/stress costs reconcile","correct per-trade costs"),
      "multiple-testing risk":item(False,"historical development record","single fixed paired test, but no untouched final interval exists","only a frozen future shadow interval can establish final validity"),
    }
    internal=all(checks[n]["verdict"]=="PASS" for n in CHECKS if n!="multiple-testing risk") and internal_no_multiple
    result={"overall":"FAIL","internal_methodology":"PASS" if internal else "FAIL","final_untouched_validity":"FAIL",
        "checks":checks,"foundation_immutable":foundation_unchanged,"all_six_timestamp_hashes_matched":hashes,"paired_identity":paired,
        "b0_reproduction":repro_ok,"same_target":paired,"same_folds_window_parameters":paired and chronology,"no_threshold_or_subset_testing":no_search,
        "s5_trade_metric_reconciliation":not trade_errors and not metric_errors,"trade_identity_reconciliation":identity_ok,
        "ranking_reconciliation":not rank_errors,"operational_artifacts_unchanged":operational,"errors":errors,
        "submitted_historical_claim":"internally valid development evidence" if internal else "invalid",
        "smallest_validation_correction":"None for internal methodology; final untouched validity requires genuinely future data after a fully qualified frozen shadow candidate." if internal else "Correct only listed validation methodology failures in a new preserved run; do not tune."}
    return result


def render(result: dict[str,Any]) -> str:
    lines=["# Independent walk-forward validation", "",f"Overall: {result['overall']}","",f"Internal methodology: {result['internal_methodology']}","",f"Final untouched validity: {result['final_untouched_validity']}","",
      "| Check | Verdict | Evidence | Failure or reason for pass | Required validation correction |","|---|---|---|---|---|"]
    for name in CHECKS:
        x=result["checks"][name];lines.append(f"| {name} | {x['verdict']} | {x['evidence']} | {x['reason']} | {x['required_validation_correction']} |")
    lines += ["","## Independent special checks","",f"- Foundation immutable: {result['foundation_immutable']}",f"- Six timestamp hashes: {result['all_six_timestamp_hashes_matched']}",
      f"- B0/B1 paired identity: {result['paired_identity']}",f"- B0 reproduction: {result['b0_reproduction']}",f"- S5 ledger/metrics: {result['s5_trade_metric_reconciliation']}",
      f"- Marginal trade identity: {result['trade_identity_reconciliation']}",f"- Ranking diagnostics: {result['ranking_reconciliation']}","",
      "All submitted metrics are historical development evidence. Final strategy validity remains FAIL regardless of internal metric quality.","",result["smallest_validation_correction"]]
    return "\n".join(lines)+"\n"


def main() -> int:
    parser=argparse.ArgumentParser();parser.add_argument("run_dir",type=Path,nargs="?");parser.add_argument("--self-check",action="store_true");args=parser.parse_args()
    if args.self_check:
        assert reward(np.array([1.,-1.]))["pf"]==1 and set(CHECKS)==set(CHECKS);print("VALIDATOR_SELF_CHECK_PASS");return 0
    if args.run_dir is None: parser.error("run_dir is required unless --self-check is used")
    run=args.run_dir.resolve();result=validate(run)
    experiment.base.write_json(run/"validator.json",result);(run/"validator.md").write_text(render(result),encoding="utf-8");shutil.copyfile(Path(__file__),run/"validator_script.py")
    manifest=experiment.base.read_json(run/"manifest.json");manifest["registry"]["validator_result"]="internal "+result["internal_methodology"]+"; final untouched FAIL"
    manifest["validator"]={"overall":result["overall"],"internal_methodology":result["internal_methodology"],"final_untouched_validity":result["final_untouched_validity"],
        "script_path":"validator_script.py","script_sha256":experiment.base.sha256(run/"validator_script.py")}
    experiment.base.add_artifact(manifest,run,run/"validator.json","independent_validator_result");experiment.base.add_artifact(manifest,run,run/"validator.md","independent_validator_report");experiment.base.add_artifact(manifest,run,run/"validator_script.py","independent_validator_script")
    experiment.base.write_json(run/"manifest.json",manifest)
    with (run/"report.md").open("a",encoding="utf-8") as handle:handle.write(f"\n## Independent validator\n\nInternal methodology: **{result['internal_methodology']}**. Final untouched validity: **FAIL**.\n")
    print(json.dumps(result,indent=2));return 0 if result["internal_methodology"]=="PASS" else 1


if __name__=="__main__":raise SystemExit(main())
