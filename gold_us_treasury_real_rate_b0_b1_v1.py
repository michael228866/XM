from __future__ import annotations
import argparse,csv,hashlib,importlib.util,json,math,subprocess
from pathlib import Path
import numpy as np,pandas as pd

EXPERIMENT="GEMINI US TREASURY REAL RATE B0 B1 V1"
OLD_RUN=Path("training_runs/20260903T071729Z_gemini_execution_aligned_label_v1")
TREASURY_RUN=Path("training_runs/20260909T074821Z_gemini_us_treasury_real_rate_foundation_v1")
TF=["UST_REAL_10Y","UST_REAL_5Y","UST_REAL_10Y_CHG_1D","UST_2S10S","UST_BREAKEVEN_10Y_PROXY"]
EXPECTED_X={
"fold1_train":"14999ac58c67bff25d7977d63a80f6069dde0256752a20a9dee936c022db69cb","fold1_score":"3c6eb5d77182428306f059e9a7d7607d1a4b5de2e5cc5217d9099e74c85e0316",
"fold2_train":"c921f11aaa44799dc798a1b54d6a745bb636b678be1212d17ba33e046acc0ac4","fold2_score":"fd77cdad26894c813e9fba0deb77ed1a5fdcaaffceaad45488da9a2015f11c36",
"fold3_train":"64124692d42cdb077fdfff1d5cf9df26582c025ed738608168b02cfa1cb42470","fold3_score":"2b55883b3410e9613983b8cebfbe1e2ea7a47daa33f44a7df535c0a5b083e717"}
EXPECTED_Y={"fold1_train":"d9bed87a073fd93c048b5a1ab8f0f7049f1a1bb8f3ff46f48a102c0b6763bbd3","fold2_train":"ff55356e8bdf1cb2cd5add50690156178ff32bc17f666f2103c8f51397fb3a12","fold3_train":"cd05d3647cd88a2fbd2b89fa3362ed5e2f9f13cbd611dee436f0e03e4a054259"}
EXPECTED_B0={"trades":689,"realized_wr":0.5660377358490566,"pf":0.8247098331219882,"mean_r":-0.07690539315994348,"pnl_r":-52.98781588720106,"cost_stress_pf":0.7838186120438296}
QUALITY={"wr":.60,"pf":1.05,"mean_r":0.0,"pnl":0.0,"stress_pf":1.0}

def args():
 p=argparse.ArgumentParser();p.add_argument("--run-dir",type=Path,required=True);return p.parse_args()
def git(root,*a): return subprocess.check_output(["git",*a],cwd=root,text=True,encoding="utf-8").strip()
def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""): h.update(b)
 return h.hexdigest()
def jwrite(p,o):
 def c(x):
  if isinstance(x,dict): return {str(k):c(v) for k,v in x.items()}
  if isinstance(x,(list,tuple)): return [c(v) for v in x]
  if isinstance(x,(np.integer,)): return int(x)
  if isinstance(x,(np.floating,float)):
   y=float(x); return y if math.isfinite(y) else None
  if isinstance(x,(np.bool_,)): return bool(x)
  if isinstance(x,pd.Timestamp): return x.isoformat()
  return x
 p.write_text(json.dumps(c(o),indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf-8")
def cwrite(p,rows):
 keys=[]
 for r in rows:
  for k in r:
   if k not in keys: keys.append(k)
 with p.open("w",newline="",encoding="utf-8") as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def loadmod(p):
 s=importlib.util.spec_from_file_location("frozen_c1",p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def metric_rows(old,trades,model):
 rows=[]; days=sum(old.core.fold_days(s,e) for _,s,e in old.FOLDS)
 for name,s,e in old.FOLDS:
  ft=[t for t in trades if s<=pd.Timestamp(t["entry_time_api"])<e]
  rows.append({"model_id":model,"fold":name,**old.trade_metrics(ft,old.core.fold_days(s,e))})
 rows.append({"model_id":model,"fold":"pooled",**old.trade_metrics(trades,days)})
 return rows

def main():
 a=args();run=a.run_dir.resolve();root=Path.cwd().resolve()
 import xgboost
 if xgboost.__version__!="3.2.0": raise RuntimeError(f"xgboost must be 3.2.0, got {xgboost.__version__}")
 if git(root,"rev-parse","HEAD")!=git(root,"rev-parse","@{u}"): raise RuntimeError("HEAD != upstream")
 manifest=json.loads((run/"manifest.json").read_text(encoding="utf-8"))
 if manifest.get("status")!="in_progress": raise RuntimeError("manifest not in_progress")
 old=loadmod(root/OLD_RUN/"training_script.py");old.drl_trading_v2.DATA_DIR=str(root)
 tz=np.load(root/TREASURY_RUN/"treasury_real_rate_features.npz")
 if list(tz["feature_names"])!=TF: raise RuntimeError("Treasury feature order mismatch")
 before={"gemini.py":sha(root/"gemini.py"),"gold_long_recent_candidate_xgb.json":sha(root/"gold_long_recent_candidate_xgb.json")}

 history,features=old.prepare_barrier_data();history=history.copy().reset_index(drop=True)
 labels=old.build_execution_aligned_labels(history);times=history["TIME_DT"];time_ns=times.to_numpy(dtype="datetime64[ns]").astype(np.int64)
 c1=labels["C1_TARGET"].to_numpy(dtype=np.int8);mature=labels["C1_MATURE"].to_numpy(dtype=bool);maturity=labels["C1_MATURITY_NS"].to_numpy(dtype=np.int64)
 lidx=np.arange(len(history),dtype=np.int64)+old.LEGACY_HORIZON_ROWS;lm=lidx<len(history);lns=np.full(len(history),np.iinfo(np.int64).max,dtype=np.int64);lns[lm]=time_ns[lidx[lm]]
 folds=[];ids=[]
 for no,(name,start,end) in enumerate(old.FOLDS,1):
  ss=int(start.to_datetime64().astype("datetime64[ns]").astype(np.int64));lower=start-pd.DateOffset(months=old.TRAIN_MONTHS)
  tr=np.flatnonzero(((times>=lower)&(times<start)&lm&mature&(lns<ss)&(maturity<ss)).to_numpy(dtype=bool))
  sc=np.flatnonzero(((times>=start)&(times<end)&mature).to_numpy(dtype=bool))
  ktr,ksc=f"fold{no}_train",f"fold{no}_score"
  xt=old.array_sha256(history.loc[tr,features].to_numpy(dtype=np.float32));xs=old.array_sha256(history.loc[sc,features].to_numpy(dtype=np.float32));yy=old.array_sha256(c1[tr])
  ok=xt==EXPECTED_X[ktr] and xs==EXPECTED_X[ksc] and yy==EXPECTED_Y[ktr]
  ids.append({"fold":name,"x_train_sha256":xt,"x_score_sha256":xs,"y_train_sha256":yy,"identity_pass":ok})
  if not ok: raise RuntimeError(f"identity gate failed {name}")
  folds.append((no,name,tr,sc,tz[ktr],tz[ksc]))
 cwrite(run/"identity_audit.csv",ids)

 models=run/"models";models.mkdir(exist_ok=True);inv=[];parts=[]
 for no,name,tr,sc,_,_ in folds:
  m=old.train_one(history,features,tr,c1[tr]);mp=models/f"B0_31_{name}_xgb.json";m.save_model(mp)
  score=old.predict_positive(m,history.loc[sc],features).astype(np.float32);inv.append({"model_id":"B0_31","fold":name,"path":mp.relative_to(run).as_posix(),"sha256":sha(mp)})
  q=history.loc[sc,["TIME_DT","OPEN","HIGH","LOW","CLOSE","ATR","M1_RSI","SPREAD"]].copy();q["global_index"]=sc;q["fold"]=name;q["fold_code"]=no-1;q["score_b0"]=score;q["score_b1"]=np.nan;parts.append(q)
 scored=pd.concat(parts,ignore_index=True)
 def execute(col,label):
  q=scored[["TIME_DT","OPEN","HIGH","LOW","CLOSE","ATR","M1_RSI","SPREAD"]].copy();q["buy_prob"]=scored[col].to_numpy(dtype=np.float32);q["sell_prob"]=np.float32(0)
  q=old.semantics.finalize_cohort(q,label,offset_hours=0);return old.semantics.simulate(q,old.semantics.SIMULATORS[-1])
 b0,a0=execute("score_b0","TREASURY_B0_31");b0r=metric_rows(old,b0,"B0_31");b0p=next(x for x in b0r if x["fold"]=="pooled")
 for k,v in EXPECTED_B0.items():
  if not math.isclose(float(b0p[k]),float(v),rel_tol=0,abs_tol=1e-12): raise RuntimeError(f"B0 reproduction failed {k}: {b0p[k]} != {v}")
 print("B0_REPRO_PASS")

 allf=features+TF
 for no,name,tr,sc,ttr,tsc in folds:
  train=history.loc[tr,features].copy();sf=history.loc[sc,features].copy()
  for j,n in enumerate(TF): train[n]=ttr[:,j];sf[n]=tsc[:,j]
  train["BARRIER_TARGET"]=c1[tr]
  m=old.train_binary_model(train,allf,1,old.N_ESTIMATORS);mp=models/f"B1_31_plus_5_treasury_{name}_xgb.json";m.save_model(mp)
  score=old.predict_positive(m,sf,allf).astype(np.float32);inv.append({"model_id":"B1_31_plus_5_treasury","fold":name,"path":mp.relative_to(run).as_posix(),"sha256":sha(mp)})
  mask=scored["fold"].eq(name).to_numpy();scored.loc[mask,"score_b1"]=score
 b1,a1=execute("score_b1","TREASURY_B1_31_PLUS_5");b1r=metric_rows(old,b1,"B1_31_plus_5_treasury");b1p=next(x for x in b1r if x["fold"]=="pooled")
 cwrite(run/"fold_metrics.csv",b0r+b1r)
 led=[]
 for mid,trades in [("B0_31",b0),("B1_31_plus_5_treasury",b1)]:
  for t in trades: z=dict(t);z["model_id"]=mid;led.append(z)
 cwrite(run/"trade_ledger.csv",led);jwrite(run/"model_inventory.json",{"models":inv,"base_features":features,"treasury_features":TF})
 np.savez_compressed(run/"paired_oof_predictions.npz",global_index=scored["global_index"].to_numpy(dtype=np.int64),time_ns=scored["TIME_DT"].to_numpy(dtype="datetime64[ns]").astype(np.int64),fold_code=scored["fold_code"].to_numpy(dtype=np.int8),score_b0=scored["score_b0"].to_numpy(dtype=np.float32),score_b1=scored["score_b1"].to_numpy(dtype=np.float32))
 delta={k:b1p[k]-b0p[k] for k in ["trades","trades_per_day","realized_wr","pf","mean_r","pnl_r","max_dd_r","cost_stress_pf"]}
 reasons=[]
 if b1p["realized_wr"]<QUALITY["wr"]: reasons.append("WR<60%")
 if b1p["pf"]<=QUALITY["pf"]: reasons.append("PF<=1.05")
 if b1p["mean_r"]<=0: reasons.append("Mean-R<=0")
 if b1p["pnl_r"]<=0: reasons.append("PnL-R<=0")
 if (b1p["break_even_adjusted_edge"] or -1)<=0: reasons.append("BE-edge<=0")
 if b1p["cost_stress_pf"]<=1: reasons.append("stress-PF<=1")
 metrics={"experiment":EXPERIMENT,"classification":"archival_reproduction_of_already_observed_frozen_B0_B1_result","b0_exact_reproduction":True,"b0":b0r,"b1":b1r,"delta_b1_minus_b0":delta,"treasury_family_status":"FAIL" if reasons else "PASS","treasury_family_failure_reasons":reasons,"treasury_direct_feature_family_closed":bool(reasons),"search_performed":False,"production_promotion_requested":False}
 jwrite(run/"metrics.json",metrics)
 report=f"""# {EXPERIMENT}

Status: **{'FAIL' if reasons else 'PASS'}**

Archival formalization of the already-observed frozen B0/B1 comparison. No post-outcome tuning, feature subset search, threshold search, lag change, transform search, weighting search, or production promotion was performed.

- B0: 31 frozen execution-aligned technical features.
- B1: same 31 plus certified Treasury 5 features.
- XGBoost 3.2.0; 220 trees; learning_rate 0.05; depth 4; min_child_weight 80; subsample/colsample 0.85; random_state 42.
- Same 18-month training windows, three chronological folds, C1 target, S5 simulator, threshold 0.75.

## B0 exact reproduction
PASS — trades={b0p['trades']}, WR={b0p['realized_wr']:.6f}, PF={b0p['pf']:.6f}, Mean-R={b0p['mean_r']:.6f}, PnL={b0p['pnl_r']:.6f}R, stress PF={b0p['cost_stress_pf']:.6f}.

## B1 Treasury
trades={b1p['trades']}, WR={b1p['realized_wr']:.6f}, PF={b1p['pf']:.6f}, Mean-R={b1p['mean_r']:.6f}, PnL={b1p['pnl_r']:.6f}R, stress PF={b1p['cost_stress_pf']:.6f}.

Delta B1-B0: {delta}

## Decision
**US TREASURY DIRECT-FEATURE FAMILY = {'FAIL' if reasons else 'PASS'}**
Failure reasons: {', '.join(reasons)}.
Stopping rule: no Treasury subset/lag/threshold/transform/interaction/weight tuning. Family closed on FAIL.
No production artifact changed.
"""
 (run/"report.md").write_text(report,encoding="utf-8")
 (run/"execution_spec.md").write_text("Frozen archival B0/B1; B0=31 technical; B1=31+5 certified Treasury; no search/tuning; S5; XGBoost 3.2.0.\n",encoding="utf-8")
 (run/"candidates.csv").write_text("candidate_id,fold,trades,trades_per_day,realized_wr,pf,mean_r,pnl,max_dd,cost_stress_pf,qualification_verdict\n"+f"B0_31,pooled,{b0p['trades']},{b0p['trades_per_day']},{b0p['realized_wr']},{b0p['pf']},{b0p['mean_r']},{b0p['pnl_r']},{b0p['max_dd_r']},{b0p['cost_stress_pf']},baseline_reproduced\n"+f"B1_31_plus_5_treasury,pooled,{b1p['trades']},{b1p['trades_per_day']},{b1p['realized_wr']},{b1p['pf']},{b1p['mean_r']},{b1p['pnl_r']},{b1p['max_dd_r']},{b1p['cost_stress_pf']},fail\n",encoding="utf-8")
 after={"gemini.py":sha(root/"gemini.py"),"gold_long_recent_candidate_xgb.json":sha(root/"gold_long_recent_candidate_xgb.json")}
 if before!=after: raise RuntimeError("operational artifact changed")
 manifest=json.loads((run/"manifest.json").read_text(encoding="utf-8"))
 manifest["data"].update({"symbols":["GOLD#","US Treasury"],"data_sources":["local XM historical exports inherited from finalized C1 identities","certified Treasury matrix from finalized Treasury foundation"],"timezone":"frozen C1 broker/API semantics; Treasury causal +2-day availability","data_start_utc":"2016-06-30T21:00:00Z","data_end_utc":"2024-12-31T20:00:00Z","train_start_utc":"2016-07-01T00:00:00","train_end_utc":"2022-12-30T19:57:00","train_rows":sum(len(x[2]) for x in folds),"validation_start_utc":"2018-01-02T00:00:00","validation_end_utc":"2024-12-31T20:00:00","validation_rows":len(scored),"test_start_utc":"not_applicable_archival_reproduction","test_end_utc":"not_applicable_archival_reproduction","test_rows":0,"purge_details":"same strict C1 parent maturity rules","embargo_details":"same as C1 parent","raw_snapshot_retained":False,"reproducibility_claim":"exact X/y identity plus finalized parent hashes; raw local CSVs not duplicated","source_files":[{"path":str(OLD_RUN/"FINALIZED.json"),"sha256":sha(root/OLD_RUN/"FINALIZED.json"),"retention_status":"retained_in_parent"},{"path":str(TREASURY_RUN/"FINALIZED.json"),"sha256":sha(root/TREASURY_RUN/"FINALIZED.json"),"retention_status":"retained_in_parent"},{"path":str(TREASURY_RUN/"treasury_real_rate_features.npz"),"sha256":sha(root/TREASURY_RUN/"treasury_real_rate_features.npz"),"retention_status":"retained_in_parent"}]})
 manifest["model"].update({"trained":True,"model_type":"paired XGBoost binary logistic chronological fold replicas","parameters":{"objective":"binary:logistic","n_estimators":220,"learning_rate":0.05,"max_depth":4,"min_child_weight":80,"subsample":0.85,"colsample_bytree":0.85,"random_state":42,"tree_method":"hist"},"boosted_rounds_or_estimators":220,"features":features,"feature_count":31,"label_definition":"C1 execution-aligned standalone S5 net realized R > 0","horizon":"90 wall-clock minutes","label_tp_sl_semantics":"same frozen C1 label","execution_tp_sl_semantics":"same frozen S5","calibration_method":"none","artifact_path":"models/","artifact_sha256":None,"retention_status":"all six fold models retained"})
 manifest["search"].update({"performed":False,"predefined_search_space":{},"candidate_results_file":"candidates.csv","not_applicable_reason":"one frozen B0/B1 comparison; no post-outcome search"})
 manifest["registry"].update({"parent_or_incumbent":"20260909T074821Z_gemini_us_treasury_real_rate_foundation_v1","selected_configuration":"B1_31_plus_5_treasury_FAIL_family_closed","trades_per_day":b1p["trades_per_day"],"realized_win_rate":b1p["realized_wr"],"pf":b1p["pf"],"mean_r":b1p["mean_r"],"pnl":b1p["pnl_r"],"max_dd":b1p["max_dd_r"],"validator_result":"PENDING"})
 manifest["promotion"].update({"requested":False,"gate_result":"not_requested_family_fail","replacement_authorized":False,"operational_artifact_changed":False})
 manifest["operational_hashes_before"]=before;manifest["operational_hashes_after"]=after;manifest["research_decision"]={"classification":"treasury_direct_feature_family_failed","family_closed":bool(reasons),"no_posthoc_tuning":True,"single_next_action":"independent validator then finalize/register FAIL"}
 jwrite(run/"manifest.json",manifest)
 print(f"B1 pooled trades={b1p['trades']} WR={b1p['realized_wr']:.6f} PF={b1p['pf']:.6f} MeanR={b1p['mean_r']:.6f} PnL={b1p['pnl_r']:.6f}")
 print("TREASURY_DIRECT_FEATURE_FAMILY_"+("FAIL" if reasons else "PASS"))
 return 0

if __name__=="__main__": raise SystemExit(main())
