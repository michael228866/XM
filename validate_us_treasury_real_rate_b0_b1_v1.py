from __future__ import annotations
import argparse,csv,hashlib,json,math
from pathlib import Path
import numpy as np

EXPECTED_B0={"trades":689,"realized_wr":0.5660377358490566,"pf":0.8247098331219882,"mean_r":-0.07690539315994348,"pnl_r":-52.98781588720106,"cost_stress_pf":0.7838186120438296}
EXPECTED_B1={"trades":3739,"realized_wr":0.4391548542391014,"pf":0.6126145620676285,"mean_r":-0.21698609871050376,"pnl_r":-811.3110230785735,"cost_stress_pf":0.5642517832482046}
def args():
 p=argparse.ArgumentParser();p.add_argument("--run-dir",type=Path,required=True);return p.parse_args()
def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def metrics(v):
 v=np.asarray(v,dtype=np.float64);w=v[v>0];l=v[v<=0];gp=float(w.sum()) if len(w) else 0.;gl=float(-l.sum()) if len(l) else 0.;pf=gp/gl if gl else (math.inf if gp else 0.)
 return {"trades":len(v),"realized_wr":float((v>0).mean()) if len(v) else 0.,"pf":pf,"mean_r":float(v.mean()) if len(v) else 0.,"pnl_r":float(v.sum())}
def stress_pf(v):
 v=np.asarray(v,dtype=np.float64);gp=float(v[v>0].sum());gl=float(-v[v<=0].sum());return gp/gl if gl else (math.inf if gp else 0.)
def close(a,b): return int(a)==int(b) if isinstance(b,int) else math.isclose(float(a),float(b),rel_tol=0,abs_tol=1e-12)
def main():
 run=args().run_dir.resolve();checks={}
 req=["manifest.json","metrics.json","report.md","fold_metrics.csv","trade_ledger.csv","identity_audit.csv","model_inventory.json","paired_oof_predictions.npz"]
 checks["required_files"]=all((run/x).is_file() for x in req)
 manifest=json.loads((run/"manifest.json").read_text(encoding="utf-8"));m=json.loads((run/"metrics.json").read_text(encoding="utf-8"))
 checks["no_search"]=manifest["search"]["performed"] is False
 checks["no_promotion"]=manifest["promotion"]["requested"] is False and manifest["promotion"]["operational_artifact_changed"] is False
 checks["family_marked_fail"]=m["treasury_family_status"]=="FAIL" and m["treasury_direct_feature_family_closed"] is True
 with (run/"identity_audit.csv").open(encoding="utf-8",newline="") as f: ids=list(csv.DictReader(f))
 checks["all_x_y_identity"]=len(ids)==3 and all(r["identity_pass"]=="True" for r in ids)
 with (run/"trade_ledger.csv").open(encoding="utf-8",newline="") as f: led=list(csv.DictReader(f))
 by={"B0_31":[],"B1_31_plus_5_treasury":[]}
 for r in led: by[r["model_id"]].append(r)
 indep={}
 for k,rows in by.items():
  net=[float(r["net_r"]) for r in rows];st=[float(r["stress_r"]) for r in rows];z=metrics(net);z["cost_stress_pf"]=stress_pf(st);indep[k]=z
 checks["b0_exact"]=all(close(indep["B0_31"][k],v) for k,v in EXPECTED_B0.items())
 checks["b1_exact_observed"]=all(close(indep["B1_31_plus_5_treasury"][k],v) for k,v in EXPECTED_B1.items())
 rb0=next(x for x in m["b0"] if x["fold"]=="pooled");rb1=next(x for x in m["b1"] if x["fold"]=="pooled")
 checks["reported_b0_matches_independent"]=all(close(rb0[k],indep["B0_31"][k]) for k in EXPECTED_B0)
 checks["reported_b1_matches_independent"]=all(close(rb1[k],indep["B1_31_plus_5_treasury"][k]) for k in EXPECTED_B1)
 inv=json.loads((run/"model_inventory.json").read_text(encoding="utf-8"))["models"]
 checks["six_models"]=len(inv)==6 and all((run/x["path"]).is_file() and sha(run/x["path"])==x["sha256"] for x in inv)
 o=np.load(run/"paired_oof_predictions.npz")
 checks["oof_complete"]=len(o["score_b0"])==2474297 and len(o["score_b1"])==2474297 and np.isfinite(o["score_b0"]).all() and np.isfinite(o["score_b1"]).all()
 checks["operational_hashes_unchanged"]=manifest.get("operational_hashes_before")==manifest.get("operational_hashes_after") and bool(manifest.get("operational_hashes_before"))
 ok=all(checks.values())
 checks={k:bool(v) for k,v in checks.items()};out={"overall":"PASS" if ok else "FAIL","checks":checks,"independent_pooled_metrics":indep}
 (run/"validator.json").write_text(json.dumps(out,indent=2,allow_nan=False)+"\n",encoding="utf-8")
 lines=["# Independent Validator","",f"Overall: **{'PASS' if ok else 'FAIL'}**","","No retraining or tuning occurs here. Economics are independently recomputed from the retained trade ledger.","","| Check | Verdict |","|---|---|"]+[f"| {k} | {'PASS' if v else 'FAIL'} |" for k,v in checks.items()]
 (run/"validator.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
 manifest["registry"]["validator_result"]="independent PASS; Treasury family FAIL" if ok else "independent FAIL";manifest["research_decision"]["validator_pass"]=ok
 (run/"manifest.json").write_text(json.dumps(manifest,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf-8")
 print("INDEPENDENT_VALIDATOR_"+("PASS" if ok else "FAIL"))
 for k,v in checks.items():print(f"{k}={'PASS' if v else 'FAIL'}")
 return 0 if ok else 1
if __name__=="__main__":raise SystemExit(main())

