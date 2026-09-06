"""Independent DATA-only validator for USD FX source reconciliation."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import training_run_history as archive

ROOT=Path(__file__).resolve().parent
PREVIOUS=ROOT/"training_runs/20260906T104638Z_gemini_usd_fx_pressure_foundation_v1"
INSTRUMENTS=("EUR/USD","GBP/USD","USD/JPY")
FEATURES=("USD_PRESSURE_1M","USD_PRESSURE_5M","USD_PRESSURE_15M","USD_PRESSURE_60M","USD_DISPERSION_15M")
HORIZONS=(1,5,15,60); SIGNS=(-1.,-1.,1.); MINUTE_NS=60_000_000_000
EXPECTED={"fold1_train":"6a7405a13f30c54e6863cf6e80ea1b2e9ee93a90902e5edd37fe4305d471aab6","fold1_score":"47086d0837f09e86d8874874de302e96e7573efb29236f24720f4a14e0e33c94","fold2_train":"691d8d2b01c3829f010ab559b1daa07c1899f8425a5b8b23b3007bf58467df44","fold2_score":"1fc6500ff642dbf7172328f71f13f38712d11f6c0ce873c38524745710c608b8","fold3_train":"3cf7f68ba2cf994d23d4db97c2e7ec10b2ed772245e3d4b776272ed20ee06b1b","fold3_score":"1451d2069b1d087dc8b4bb7b3ed5840a7b2c03d4adfb548eadc611ed07803449"}


def digest(values:np.ndarray)->str:
    values=np.ascontiguousarray(values); h=hashlib.sha256(); h.update(str(values.dtype).encode()); h.update(np.asarray(values.shape,dtype=np.int64).tobytes()); h.update(values.tobytes()); return h.hexdigest()


def inventory(path:Path)->dict[str,str]:
    return {p.relative_to(path).as_posix():archive.file_sha256(p) for p in sorted(path.rglob("*")) if p.is_file()}


def asof(opens:np.ndarray,prices:np.ndarray,anchors:np.ndarray)->tuple[np.ndarray,np.ndarray]:
    available=opens+MINUTE_NS; idx=np.searchsorted(available,anchors,side="right")-1; exists=idx>=0; safe=np.maximum(idx,0); age=np.where(exists,anchors-available[safe],np.iinfo(np.int64).max); fresh=exists&(age>=0)&(age<=5*MINUTE_NS); return np.where(fresh,prices[safe],np.nan),fresh


def within(values:np.ndarray,intervals:list[tuple[int,int]])->np.ndarray:
    if not intervals:return np.zeros(len(values),dtype=bool)
    merged=[]
    for start,end in sorted(intervals):
        if merged and start<=merged[-1][1]:merged[-1][1]=max(merged[-1][1],end)
        else:merged.append([start,end])
    starts=np.asarray([x[0] for x in merged],dtype=np.int64); ends=np.asarray([x[1] for x in merged],dtype=np.int64); idx=np.searchsorted(starts,values,side="right")-1; valid=idx>=0; result=np.zeros(len(values),dtype=bool); result[valid]=values[valid]<ends[idx[valid]]; return result


def reconstruct(run:Path,times:np.ndarray,provider:str)->tuple[np.ndarray,np.ndarray,np.ndarray]:
    gaps=pd.read_csv(run/"source_gap_audit.csv"); closures={x:[] for x in INSTRUMENTS}
    for row in gaps.itertuples():
        if row.classification!="unexplained_source_gap": closures[row.instrument].append((pd.Timestamp(row.gap_start_utc).value+MINUTE_NS,pd.Timestamp(row.gap_end_utc).value+MINUTE_NS))
    component=np.full((len(times),3,4),np.nan); unknown_c=np.zeros_like(component,dtype=bool)
    for i,economic in enumerate(INSTRUMENTS):
        prefix="DUKASCOPY_" if provider=="Dukascopy" else ""
        with np.load(run/f"fx_source_{prefix}{re.sub('[^A-Z]','',economic)}.npz",allow_pickle=False) as source:
            opens=source["open_utc_ns"].astype(np.int64); close=source["close"].astype(float)
        current,current_ok=asof(opens,close,times)
        for j,horizon in enumerate(HORIZONS):
            anchor=times-horizon*MINUTE_NS; past,past_ok=asof(opens,close,anchor); valid=current_ok&past_ok
            component[:,i,j]=SIGNS[i]*np.log(current/past)
            legitimate=(~valid)&((current_ok|within(times,closures[economic]))&(past_ok|within(anchor,closures[economic])))
            unknown_c[:,i,j]=(~valid)&~legitimate
    matrix=np.full((len(times),5),np.nan); unknown=np.zeros_like(matrix,dtype=bool); legitimate=np.zeros_like(matrix,dtype=bool)
    for j in range(4):
        ok=np.isfinite(component[:,:,j]).all(axis=1); matrix[ok,j]=component[ok,:,j].mean(axis=1); unknown[:,j]=unknown_c[:,:,j].any(axis=1); legitimate[:,j]=(~ok)&~unknown[:,j]
    ok=np.isfinite(component[:,:,2]).all(axis=1); matrix[ok,4]=component[ok,:,2].std(axis=1,ddof=1); unknown[:,4]=unknown_c[:,:,2].any(axis=1); legitimate[:,4]=(~ok)&~unknown[:,4]
    return matrix,unknown,legitimate


def validate(run:Path)->int:
    manifest=archive.read_json(run/"manifest.json"); metrics=archive.read_json(run/"metrics.json"); attempts=archive.read_json(run/"source_attempts.json")
    checks=[]
    def check(name:str,condition:bool,evidence:Any,category:str="methodology")->None:checks.append({"check":name,"verdict":"PASS" if condition else "FAIL","evidence":evidence,"category":category})
    hierarchy=list(attempts)==["XM","Dukascopy","TrueFX"] and attempts["XM"]["attempted"]
    first_pass=not(attempts["XM"]["certification"]=="PASS" and attempts["Dukascopy"]["attempted"]) and not(attempts["Dukascopy"]["certification"]=="PASS" and attempts["TrueFX"]["attempted"])
    check("source hierarchy followed",hierarchy and first_pass,attempts)
    provider=metrics.get("certified_provider"); count=metrics.get("source_provider_count")
    reconstruction_provider=provider or ("Dukascopy" if attempts["Dukascopy"]["attempted"] and (run/"fx_source_DUKASCOPY_EURUSD.npz").is_file() else None)
    check("no provider splicing and same provider for all pairs",(provider in ("XM","Dukascopy") and count==1) or (provider is None and count==0),{"provider":provider,"count":count})
    check("no outcome-based provider selection",manifest["foundation_specification"]["selection_basis"]=="provenance, completeness, timestamps and causality only",manifest["foundation_specification"]["selection_basis"])
    with np.load(run/"exact_timestamps.npz",allow_pickle=False) as exact:
        parts=[]; details={}; exact_ok=True
        for block,expected in EXPECTED.items():
            broker=exact[block+"_broker_ns"].astype(np.int64); utc=exact[block+"_utc_ns"].astype(np.int64); exact_ok&=digest(broker)==expected; parts.append(utc); details[block]=digest(broker)
        times=np.unique(np.concatenate(parts)).astype(np.int64)
    check("six frozen timestamp hashes",exact_ok,details)
    source_hashes_ok=True; source_rows={}
    if reconstruction_provider:
        for economic in INSTRUMENTS:
            prefix="DUKASCOPY_" if reconstruction_provider=="Dukascopy" else ""; path=run/f"fx_source_{prefix}{re.sub('[^A-Z]','',economic)}.npz"
            source_hashes_ok&=path.is_file()
            if path.is_file():
                with np.load(path,allow_pickle=False) as source:
                    opens=source["open_utc_ns"].astype(np.int64); vals=np.column_stack([source[x] for x in ("open","high","low","close")]); source_rows[economic]=len(opens)
                    source_hashes_ok&=len(opens)>0 and bool(np.all(np.diff(opens)>0)) and bool(np.isfinite(vals).all()) and bool((vals>0).all())
    check("historical completeness and source hashes",source_hashes_ok and metrics["unresolved_source_rows_final"]==0,{"rows":source_rows,"unresolved":metrics["unresolved_source_rows_final"]},"data")
    gaps=pd.read_csv(run/"source_gap_audit.csv"); unexplained=int((gaps["classification"]=="unexplained_source_gap").sum()) if len(gaps) else 0
    check("closure classification evidence",unexplained==0,{"unexplained_gap_count":unexplained},"data")
    with np.load(run/"usd_fx_reconciled_feature_matrix.npz",allow_pickle=False) as saved:
        saved_times=saved["utc_ns"].astype(np.int64); saved_matrix=saved["features"].astype(float); names=saved["feature_names"].astype(str).tolist(); saved_unknown=saved["unknown_source_mask"].astype(bool); saved_legit=saved["legitimate_nan_mask"].astype(bool)
    rebuilt,unknown,legitimate=reconstruct(run,times,reconstruction_provider) if reconstruction_provider else (np.full_like(saved_matrix,np.nan),np.ones_like(saved_unknown),np.zeros_like(saved_legit))
    matrix_ok=np.array_equal(saved_times,times) and names==list(FEATURES) and np.array_equal(saved_matrix,rebuilt,equal_nan=True)
    check("causal completed-bar and staleness",matrix_ok and np.array_equal(saved_unknown,unknown) and np.array_equal(saved_legit,legitimate),{"max_staleness_minutes":5})
    check("return horizons and sign orientation",matrix_ok,{"horizons":HORIZONS,"signs":SIGNS})
    check("equal mean and ddof=1 dispersion",matrix_ok,names)
    check("no interpolation or unlimited forward fill",not np.any(np.isfinite(saved_matrix)&saved_unknown),{"unknown_rows":int(unknown.any(axis=1).sum())})
    check("new matrix identity",digest(saved_matrix)==metrics["usd_fx_reconciled_feature_matrix_sha256"] and digest(saved_matrix)!="da226ee13e8470a919cc4b08770e76c376911e8d4613e0bc5ffc98a8395fdf81",digest(saved_matrix))
    protected=manifest["protected_runs_before"]; previous_ok=PREVIOUS.name in protected and inventory(PREVIOUS)==protected[PREVIOUS.name]
    check("previous failed run byte-identical",previous_ok,PREVIOUS.name)
    current={name:archive.file_sha256(ROOT/name) for name in manifest["operational_hashes_before"]}
    check("operational artifacts unchanged",current==manifest["operational_hashes_before"]==manifest["operational_hashes_after"],current)
    source=(run/manifest["training_script_snapshot"]).read_text(encoding="utf-8").lower(); forbidden=[x for x in ("import xgboost","import sklearn","predict_proba","trade_ledger","realized_r") if x in source]
    check("no model label outcome or performance access",not forbidden and not metrics["model_training_performed"] and not metrics["strategy_evaluation_performed"],forbidden)
    method_pass=all(x["verdict"]=="PASS" for x in checks if x["category"]=="methodology"); data_pass=all(x["verdict"]=="PASS" for x in checks); ready=bool(method_pass and data_pass and metrics["data_foundation_ready"])
    result={"internal_methodology":"PASS" if method_pass else "FAIL","data_certification":"PASS" if data_pass else "FAIL","data_foundation_ready":ready,"checks":checks,"model_training_performed":False,"strategy_evaluation_performed":False}
    archive.write_json(run/"validator.json",result)
    lines=["# Independent DATA-only validator","",f"Internal methodology: **{result['internal_methodology']}**","",f"Data certification: **{result['data_certification']}**","",f"Foundation ready: **{'YES' if ready else 'NO'}**","","| Check | Verdict | Evidence |","|---|---|---|"]+[f"| {x['check']} | {x['verdict']} | {str(x['evidence']).replace('|','/')} |" for x in checks]
    (run/"validator.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    metrics.update({"validator_internal_methodology":result["internal_methodology"],"validator_data_certification":result["data_certification"],"data_foundation_ready":ready,"run_status":"pass" if ready else "fail"}); archive.write_json(run/"metrics.json",metrics)
    report=(run/"report.md").read_text(encoding="utf-8").split("\n## Independent validation\n",1)[0].rstrip()+f"\n\n## Independent validation\n\nInternal methodology: **{result['internal_methodology']}**. Data certification: **{result['data_certification']}**. Foundation ready: **{'YES' if ready else 'NO'}**.\n"; (run/"report.md").write_text(report,encoding="utf-8")
    manifest["registry"]["validator_result"]=f"internal {result['internal_methodology']}; data certification {result['data_certification']}"
    for name in ("metrics.json","report.md","validator.json","validator.md","validator_script.py"):
        manifest["artifacts"]=[x for x in manifest["artifacts"] if x.get("path")!=name]; manifest["artifacts"].append({"kind":"validator_evidence","path":name,"sha256":archive.file_sha256(run/name),"retention_status":"stored_in_finalized_run_git_archival_pending"})
    archive.write_json(run/"manifest.json",manifest); print(json.dumps(result,indent=2)); return 0 if method_pass else 1


def self_check()->None:
    values,fresh=asof(np.array([0,MINUTE_NS]),np.array([1.,2.]),np.array([MINUTE_NS,2*MINUTE_NS])); assert fresh.all() and values.tolist()==[1.,2.]; print("SOURCE_RECONCILIATION_VALIDATOR_SELF_CHECK_PASS")


if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("run_dir",type=Path,nargs="?"); p.add_argument("--self-check",action="store_true"); a=p.parse_args()
    if a.self_check:self_check()
    elif a.run_dir is None:p.error("run_dir required")
    else:raise SystemExit(validate(a.run_dir.resolve()))
