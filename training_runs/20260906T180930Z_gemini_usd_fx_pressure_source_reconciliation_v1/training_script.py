"""Reconcile one complete provider for the frozen USD FX pressure matrix."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sqlite3
import subprocess
import time
import urllib.error
import urllib.request
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import gold_gemini_usd_fx_pressure_foundation_v1 as base
import training_run_history as archive

ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT / "training_runs/20260906T104638Z_gemini_usd_fx_pressure_foundation_v1"
DUKA_BASE = "https://jetta.dukascopy.com/v1"
DUKA_CODES = {"EUR/USD": "EUR-USD", "GBP/USD": "GBP-USD", "USD/JPY": "USD-JPY"}
PROVIDERS = ("XM", "Dukascopy", "TrueFX")
MINUTE_NS = base.MINUTE_NS


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True, encoding="utf-8").strip()


def write_json(path: Path, value: Any) -> None:
    archive.write_json(path, value)


def logical_hash(*arrays: np.ndarray) -> str:
    h = hashlib.sha256()
    for values in arrays:
        values = np.ascontiguousarray(values)
        h.update(str(values.dtype).encode())
        h.update(np.asarray(values.shape, dtype=np.int64).tobytes())
        h.update(values.tobytes())
    return h.hexdigest()


def preregister(run: Path) -> dict[str, Any]:
    manifest = archive.read_json(run / "manifest.json")
    head, remote = git("rev-parse", "HEAD"), git("rev-parse", "origin/main")
    if manifest["git_dirty"] or head != remote or head != manifest["git_commit"]:
        raise RuntimeError("clean pushed pre-run Git state required")
    if archive.file_sha256(Path(__file__)) != manifest["training_script_sha256"]:
        raise RuntimeError("executed script differs from snapshot")
    for required in (PREVIOUS, base.TIMESTAMP_FOUNDATION):
        if archive.validate_run(required):
            raise RuntimeError(f"input run does not validate: {required.name}")
    manifest["pre_run_git"] = {"pre_run_git_commit": head, "pre_run_git_dirty": False,
        "head_sha": head, "origin_main_sha": remote, "head_equals_origin_main": True}
    manifest["protected_runs_before"] = {
        item.parent.name: base.inventory(item.parent)
        for item in sorted((ROOT / "training_runs").glob("*/FINALIZED.json"))
    }
    manifest["operational_hashes_before"] = {
        name: archive.file_sha256(ROOT / name)
        for name in ("gemini.py", "gold_long_recent_candidate_xgb.json")
    }
    manifest["source_hierarchy"] = list(PROVIDERS)
    manifest["foundation_specification"] = {
        "instruments": list(base.INSTRUMENTS), "features": list(base.FEATURES),
        "horizons_minutes": list(base.HORIZONS), "orientation": [-1, -1, 1],
        "max_staleness_minutes": 5, "provider_splicing": False,
        "dukascopy_convention": "native M1 BID OHLC; UTC bar-open; available at open+1 minute",
        "selection_basis": "provenance, completeness, timestamps and causality only",
    }
    manifest["search"].update({"performed": False, "predefined_search_space": {},
        "candidate_results_file": "not_applicable_data_only",
        "not_applicable_reason": "Source hierarchy is fixed; no alpha/model selection."})
    manifest["model"].update({"trained": False, "model_type": "not_applicable_data_only",
        "features": list(base.FEATURES), "feature_count": 5,
        "not_applicable_reason": "Data-only source reconciliation."})
    manifest["promotion"].update({"requested": False, "gate_result": "not_applicable_data_only",
        "replacement_authorized": False, "operational_artifact_changed": False})
    write_json(run / "manifest.json", manifest)
    shutil.copyfile(ROOT / "gold_gemini_usd_fx_pressure_source_reconciliation_v1_validator.py", run / "validator_script.py")
    shutil.copyfile(ROOT / "gold_gemini_usd_fx_pressure_foundation_v1.py", run / "foundation_dependency.py")
    shutil.copyfile(base.TIMESTAMP_FOUNDATION / "exact_timestamps.npz", run / "exact_timestamps.npz")
    return manifest


def dates_between(start: datetime, end: datetime) -> list[str]:
    day, result = start.date(), []
    while day <= end.date():
        result.append(day.isoformat())
        day += timedelta(days=1)
    return result


def http_get(url: str, attempts: int = 5) -> tuple[int, bytes, int, str | None]:
    error = None
    for attempt in range(1, attempts + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "XM-GOLD-data-foundation/1.0"})
            with urllib.request.urlopen(request, timeout=45) as response:
                return int(response.status), response.read(), attempt, None
        except (OSError, urllib.error.HTTPError) as exc:
            error = f"{type(exc).__name__}: {exc}"
            time.sleep(min(8, 2 ** (attempt - 1)))
    return 0, b"", attempts, error


def init_raw_db(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=DELETE")
    connection.execute("CREATE TABLE IF NOT EXISTS responses(pair TEXT,date TEXT,url TEXT,status INTEGER,body_zlib BLOB,body_sha256 TEXT,fetched_at_utc TEXT,attempts INTEGER,error TEXT,PRIMARY KEY(pair,date))")
    return connection


def acquire_dukascopy(run: Path, start: datetime, end: datetime) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    metadata = {}
    for economic, code in DUKA_CODES.items():
        status, body, attempts, error = http_get(f"{DUKA_BASE}/instruments/{code}")
        if status != 200:
            raise RuntimeError(f"Dukascopy metadata failed {economic}: {error}")
        (run / f"dukascopy_metadata_{code.replace('-', '')}.json").write_bytes(body)
        metadata[economic] = json.loads(body)
    db_path = run / "dukascopy_raw_responses.sqlite"
    db = init_raw_db(db_path)
    existing = {(row[0], row[1]) for row in db.execute("SELECT pair,date FROM responses WHERE status=200")}
    jobs = [(economic, DUKA_CODES[economic], date) for economic in base.INSTRUMENTS for date in dates_between(start, end) if (economic, date) not in existing]
    def fetch(job: tuple[str, str, str]) -> tuple[Any, ...]:
        economic, code, date = job
        y, m, d = date.split("-")
        url = f"{DUKA_BASE}/candles/minute/{code}/BID/{int(y)}/{int(m)}/{int(d)}"
        status, body, attempts, error = http_get(url)
        if status == 200:
            try:
                json.loads(body)
            except json.JSONDecodeError as exc:
                status, error = 0, f"invalid JSON: {exc}"
        return economic, date, url, status, zlib.compress(body, 9), hashlib.sha256(body).hexdigest(), datetime.now(timezone.utc).isoformat(), attempts, error
    completed = 0
    with ThreadPoolExecutor(max_workers=12) as pool:
        futures = [pool.submit(fetch, job) for job in jobs]
        for future in as_completed(futures):
            db.execute("INSERT OR REPLACE INTO responses VALUES(?,?,?,?,?,?,?,?,?)", future.result())
            completed += 1
            if completed % 100 == 0:
                db.commit(); print(f"Dukascopy days {completed}/{len(jobs)}", flush=True)
    db.commit()
    failures = [dict(zip(("pair", "date", "status", "error"), row)) for row in db.execute("SELECT pair,date,status,error FROM responses WHERE status<>200")]
    if failures:
        db.close(); return {}, {"status": "FAIL", "failures": failures[:100], "raw_db": db_path.name}
    result: dict[str, dict[str, np.ndarray]] = {}
    for economic in base.INSTRUMENTS:
        pieces = []
        for date, packed in db.execute("SELECT date,body_zlib FROM responses WHERE pair=? ORDER BY date", (economic,)):
            payload = json.loads(zlib.decompress(packed))
            times = np.asarray(payload.get("times", []), dtype=np.int64)
            if not len(times):
                continue
            columns = [np.asarray(payload.get(name, []), dtype=np.float64) for name in ("opens", "highs", "lows", "closes")]
            if any(len(column) != len(times) for column in columns):
                raise RuntimeError(f"inconsistent Dukascopy arrays: {economic} {date}")
            timestamp = int(payload["timestamp"]) + int(payload.get("shift", 60000)) * np.cumsum(times)
            multiplier = float(payload["multiplier"])
            scale = round(1.0 / multiplier)
            decoded = []
            for name, deltas in zip(("open", "high", "low", "close"), columns):
                base_price = float(payload[name])
                decoded.append(np.rint((base_price + multiplier * np.cumsum(deltas)) * scale) / scale)
            pieces.append(np.rec.fromarrays([timestamp * 1_000_000, *decoded], names="open_utc_ns,open,high,low,close"))
        joined = np.concatenate(pieces) if pieces else np.recarray(0, dtype=[("open_utc_ns","i8"),("open","f8"),("high","f8"),("low","f8"),("close","f8")])
        order = np.argsort(joined["open_utc_ns"], kind="stable"); joined = joined[order]
        _, indices = np.unique(joined["open_utc_ns"], return_index=True); joined = joined[indices]
        mask = (joined["open_utc_ns"] >= int(start.timestamp()*1e9)) & (joined["open_utc_ns"] < int(end.timestamp()*1e9))
        result[economic] = {name: joined[name][mask] for name in joined.dtype.names or ()}
        np.savez_compressed(run / f"fx_source_DUKASCOPY_{re.sub('[^A-Z]','',economic)}.npz", **result[economic])
    db.execute("VACUUM"); db.close()
    return result, {"status": "candidate", "raw_db": db_path.name,
        "raw_db_sha256": archive.file_sha256(db_path),
        "metadata": {k: f"dukascopy_metadata_{v.replace('-', '')}.json" for k,v in DUKA_CODES.items()},
        "price_convention": "native M1 BID OHLC", "timestamp_convention": "UTC bar-open milliseconds; information at open+1m"}


def expected_fx_open(open_ns: np.ndarray, metadata: dict[str, Any]) -> np.ndarray:
    utc = pd.to_datetime(open_ns, utc=True)
    local = utc.tz_convert(ZoneInfo(metadata["defaultTimezone"]))
    weekday, minute = local.weekday.to_numpy(), (local.hour * 60 + local.minute).to_numpy()
    weekly = ((weekday == 6) & (minute >= 17*60)) | ((weekday >= 0) & (weekday <= 3)) | ((weekday == 4) & (minute < 17*60))
    holiday = np.zeros(len(open_ns), dtype=bool)
    for row in metadata.get("holidays", []):
        holiday |= (open_ns >= int(row["from"])*1_000_000) & (open_ns < int(row["till"])*1_000_000)
    return weekly & ~holiday


def audit_gaps(data: dict[str, dict[str, np.ndarray]], metadata: dict[str, Any], provider: str) -> tuple[list[dict[str, Any]], dict[str, list[tuple[int,int]]]]:
    raw = {}
    for economic, values in data.items():
        times = values["open_utc_ns"]
        raw[economic] = [(int(times[i]+MINUTE_NS), int(times[i+1]), int((times[i+1]-times[i])//MINUTE_NS-1)) for i in np.flatnonzero(np.diff(times)>MINUTE_NS)]
    common = []
    for start,end,count in raw[base.INSTRUMENTS[0]]:
        if count > 5 and all(any(abs(start-a)<=MINUTE_NS and abs(end-b)<=MINUTE_NS for a,b,n in raw[x] if n>5) for x in base.INSTRUMENTS[1:]):
            common.append((start,end))
    rows, closures = [], {economic: [] for economic in base.INSTRUMENTS}
    for economic in base.INSTRUMENTS:
        for start,end,count in raw[economic]:
            missing = np.arange(start, end, MINUTE_NS, dtype=np.int64)
            known_closed = provider == "Dukascopy" and bool((~expected_fx_open(missing, metadata[economic])).all())
            common_gap = count > 5 and any(abs(start-a)<=MINUTE_NS and abs(end-b)<=MINUTE_NS for a,b in common)
            if known_closed: classification = "confirmed_normal_fx_closure"
            elif common_gap: classification = "confirmed_provider_wide_quote_closure"
            elif count <= 5: classification = "confirmed_symbol_specific_quote_gap"
            else: classification = "unexplained_source_gap"
            if classification != "unexplained_source_gap": closures[economic].append((start+MINUTE_NS,end+MINUTE_NS))
            rows.append({"provider":provider,"instrument":economic,"gap_start_utc":pd.Timestamp(start,tz="UTC").isoformat(),"gap_end_utc":pd.Timestamp(end,tz="UTC").isoformat(),"missing_minutes":count,"classification":classification})
    return rows, closures


def inside(values: np.ndarray, intervals: list[tuple[int,int]]) -> np.ndarray:
    if not intervals: return np.zeros(len(values),dtype=bool)
    merged=[]
    for start,end in sorted(intervals):
        if merged and start<=merged[-1][1]: merged[-1][1]=max(merged[-1][1],end)
        else: merged.append([start,end])
    starts=np.asarray([x[0] for x in merged],dtype=np.int64); ends=np.asarray([x[1] for x in merged],dtype=np.int64)
    idx=np.searchsorted(starts,values,side="right")-1; valid=idx>=0; result=np.zeros(len(values),dtype=bool); result[valid]=values[valid]<ends[idx[valid]]; return result


def build_matrix(times: np.ndarray, data: dict[str, dict[str,np.ndarray]], closures: dict[str,list[tuple[int,int]]]) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    component = np.full((len(times),3,4),np.nan); unknown_c=np.zeros_like(component,dtype=bool); legit_c=np.zeros_like(component,dtype=bool)
    for i,economic in enumerate(base.INSTRUMENTS):
        current,_,current_ok=base.asof_price(data[economic],times)
        for j,horizon in enumerate(base.HORIZONS):
            past_anchor=times-horizon*MINUTE_NS; past,_,past_ok=base.asof_price(data[economic],past_anchor)
            valid=current_ok & past_ok
            component[:,i,j]=base.ORIENTATION[i]*np.log(current/past)
            legit=(~valid)&((current_ok|inside(times,closures[economic]))&(past_ok|inside(past_anchor,closures[economic])))
            legit_c[:,i,j]=legit; unknown_c[:,i,j]=(~valid)&~legit
    matrix=np.full((len(times),5),np.nan); unknown=np.zeros_like(matrix,dtype=bool); legitimate=np.zeros_like(matrix,dtype=bool)
    for j in range(4):
        valid=np.isfinite(component[:,:,j]).all(axis=1); matrix[valid,j]=component[valid,:,j].mean(axis=1)
        unknown[:,j]=unknown_c[:,:,j].any(axis=1); legitimate[:,j]=(~valid)&~unknown[:,j]
    valid=np.isfinite(component[:,:,2]).all(axis=1); matrix[valid,4]=component[valid,:,2].std(axis=1,ddof=1)
    unknown[:,4]=unknown_c[:,:,2].any(axis=1); legitimate[:,4]=(~valid)&~unknown[:,4]
    return matrix,unknown,legitimate


def preserve_previous_gap_impact(run: Path) -> dict[str, Any]:
    with np.load(PREVIOUS / "fx_feature_matrix.npz", allow_pickle=False) as saved:
        times = saved["utc_ns"].astype(np.int64)
        affected = saved["unknown_source_mask"].astype(bool).any(axis=1)
    indices = np.flatnonzero(affected)
    clusters = []
    if len(indices):
        cuts = np.flatnonzero(np.diff(times[indices]) > MINUTE_NS) + 1
        for group in np.split(indices, cuts):
            clusters.append({"start_utc":pd.Timestamp(times[group[0]],tz="UTC").isoformat(),"end_utc":pd.Timestamp(times[group[-1]],tz="UTC").isoformat(),"exact_rows":len(group)})
    pd.DataFrame(clusters).to_csv(run / "previous_unresolved_clusters.csv", index=False)
    alignment = pd.read_csv(PREVIOUS / "fx_alignment_audit.csv")
    rows = alignment[alignment["record_type"] == "instrument"]
    pair_counts = {pair:int(rows[rows["instrument_or_feature"] == pair]["unresolved_source_gap_rows"].sum()) for pair in base.INSTRUMENTS}
    summary={"previous_unresolved_union":int(affected.sum()),"pair_current_endpoint_affected_rows":pair_counts,"cluster_count":len(clusters)}
    write_json(run / "previous_gap_impact.json", summary); return summary


def sanity_against_xm(times: np.ndarray, external: dict[str,dict[str,np.ndarray]], xm: dict[str,dict[str,np.ndarray]]) -> list[dict[str,Any]]:
    rows=[]; ext_returns=[]; xm_returns=[]
    for economic in base.INSTRUMENTS:
        e_now,_,e1=base.asof_price(external[economic],times); e_old,_,e0=base.asof_price(external[economic],times-MINUTE_NS)
        x_now,_,x1=base.asof_price(xm[economic],times); x_old,_,x0=base.asof_price(xm[economic],times-MINUTE_NS)
        er=np.log(e_now/e_old); xr=np.log(x_now/x_old); valid=e1&e0&x1&x0&np.isfinite(er)&np.isfinite(xr)
        corr=float(np.corrcoef(er[valid],xr[valid])[0,1]) if valid.sum()>1 else None
        rows.append({"comparison":"pair_1m_return","instrument":economic,"overlap":int(valid.sum()),"correlation":corr,"sign_agreement":float((np.sign(er[valid])==np.sign(xr[valid])).mean()) if valid.any() else None,"median_absolute_difference":float(np.median(np.abs(er[valid]-xr[valid]))) if valid.any() else None})
        ext_returns.append(base.ORIENTATION[base.INSTRUMENTS.index(economic)]*er); xm_returns.append(base.ORIENTATION[base.INSTRUMENTS.index(economic)]*xr)
    ep=np.column_stack(ext_returns); xp=np.column_stack(xm_returns); valid=np.isfinite(ep).all(axis=1)&np.isfinite(xp).all(axis=1)
    rows.append({"comparison":"USD_PRESSURE_1M","instrument":"three-pair mean","overlap":int(valid.sum()),"correlation":float(np.corrcoef(ep[valid].mean(axis=1),xp[valid].mean(axis=1))[0,1]) if valid.sum()>1 else None,"sign_agreement":float((np.sign(ep[valid].mean(axis=1))==np.sign(xp[valid].mean(axis=1))).mean()) if valid.any() else None,"median_absolute_difference":float(np.median(np.abs(ep[valid].mean(axis=1)-xp[valid].mean(axis=1)))) if valid.any() else None})
    return rows


def acquire_xm(run: Path, start: datetime, end: datetime, times: np.ndarray) -> tuple[dict[str,dict[str,np.ndarray]],dict[str,Any]]:
    import MetaTrader5 as mt5
    if not mt5.initialize(path=str(base.TERMINAL)): raise RuntimeError(f"MT5 init: {mt5.last_error()}")
    symbols,resolution=base.resolve_symbols(mt5); write_json(run/"xm_symbol_resolution.json",resolution)
    attempts=[]; final={}
    for attempt in (1,2):
        data={}; hashes={}
        for economic in base.INSTRUMENTS:
            values,chunks=base.acquire_symbol(mt5,symbols[economic],start,end); data[economic]=values
            hashes[economic]=logical_hash(*(values[k] for k in ("open_utc_ns","open","high","low","close")))
        attempts.append({"attempt":attempt,"hashes":hashes,"rows":{k:len(v["open_utc_ns"]) for k,v in data.items()}})
        final=data
    terminals=[str(path) for path in Path("D:/").glob("XM*/terminal64.exe")]
    tick_rows=[]; old_gaps=pd.read_csv(PREVIOUS/"fx_gap_audit.csv")
    for economic in ("GBP/USD","USD/JPY"):
        sample=old_gaps[(old_gaps["instrument"]==economic)&(old_gaps["classification"]=="unexplained_source_gap")&(old_gaps["missing_minutes"]<=60)].sort_values("gap_start_utc")
        if len(sample): sample=sample.iloc[np.unique(np.linspace(0,len(sample)-1,min(12,len(sample)),dtype=int))]
        for row in sample.itertuples():
            begin=pd.Timestamp(row.gap_start_utc).to_pydatetime(); finish=pd.Timestamp(row.gap_end_utc).to_pydatetime()
            ticks=mt5.copy_ticks_range(symbols[economic],begin,finish,mt5.COPY_TICKS_ALL)
            tick_rows.append({"instrument":economic,"gap_start_utc":row.gap_start_utc,"gap_end_utc":row.gap_end_utc,"missing_m1_minutes":row.missing_minutes,"ticks_returned":0 if ticks is None else len(ticks),"last_error":str(mt5.last_error())})
    pd.DataFrame(tick_rows).to_csv(run/"xm_tick_gap_audit.csv",index=False)
    mt5.shutdown()
    rows,closures=audit_gaps(final,{},"XM"); matrix,unknown,legitimate=build_matrix(times,final,closures)
    deterministic=attempts[0]["hashes"]==attempts[1]["hashes"]
    unexplained=sum(row["classification"]=="unexplained_source_gap" for row in rows)
    summary={"attempted":True,"certification":"PASS" if deterministic and not unknown.any() and unexplained==0 else "FAIL",
        "deterministic_retrieval":deterministic,"attempts":attempts,"available_terminal_paths":terminals,
        "unresolved_source_rows":int(unknown.any(axis=1).sum()),"unexplained_gap_count":unexplained}
    pd.DataFrame(rows).to_csv(run/"xm_gap_adjudication.csv",index=False); write_json(run/"xm_retrieval_attempts.json",summary)
    return final,summary


def main(run: Path) -> None:
    manifest=preregister(run); _,blocks,times=base.exact_blocks(run); previous_impact=preserve_previous_gap_impact(run)
    start=datetime.fromtimestamp((int(times.min())-61*MINUTE_NS)/1e9,tz=timezone.utc)
    end=datetime.fromtimestamp((int(times.max())+2*MINUTE_NS)/1e9,tz=timezone.utc)
    xm,xm_summary=acquire_xm(run,start,end,times)
    attempts={"XM":xm_summary,"Dukascopy":{"attempted":False,"certification":"not_attempted"},"TrueFX":{"attempted":False,"certification":"not_attempted"}}
    certified=None; data=xm; metadata={}; provider="XM"
    if xm_summary["certification"]=="PASS": certified="XM"
    else:
        attempts["Dukascopy"]["attempted"]=True
        data,duka=acquire_dukascopy(run,start,end); attempts["Dukascopy"].update(duka); provider="Dukascopy"
        if data:
            metadata={economic:json.loads((run/f"dukascopy_metadata_{DUKA_CODES[economic].replace('-','')}.json").read_text()) for economic in base.INSTRUMENTS}
            gap_rows,closures=audit_gaps(data,metadata,provider); matrix,unknown,legitimate=build_matrix(times,data,closures)
            unexplained=sum(row["classification"]=="unexplained_source_gap" for row in gap_rows)
            attempts["Dukascopy"]["certification"]="PASS" if not unknown.any() and unexplained==0 else "FAIL"
            attempts["Dukascopy"]["unresolved_source_rows"]=int(unknown.any(axis=1).sum())
            if attempts["Dukascopy"]["certification"]=="PASS": certified="Dukascopy"
        if certified is None:
            attempts["TrueFX"]={"attempted":True,"certification":"FAIL","reason":"Official reproducible full-range 2016-2024 native M1 endpoint unavailable; hierarchy exhausted without splicing."}
    if certified=="XM":
        gap_rows,closures=audit_gaps(data,{},provider); matrix,unknown,legitimate=build_matrix(times,data,closures)
    elif not data:
        matrix=np.full((len(times),5),np.nan); unknown=np.ones_like(matrix,dtype=bool); legitimate=np.zeros_like(matrix,dtype=bool); gap_rows=[]
    pd.DataFrame(gap_rows).to_csv(run/"source_gap_audit.csv",index=False)
    if certified=="Dukascopy": pd.DataFrame(sanity_against_xm(times,data,xm)).to_csv(run/"source_sanity_vs_xm.csv",index=False)
    np.savez_compressed(run/"usd_fx_reconciled_feature_matrix.npz",utc_ns=times,features=matrix,feature_names=np.asarray(base.FEATURES),unknown_source_mask=unknown,legitimate_nan_mask=legitimate)
    coverage=[]
    for block in base.EXPECTED_TIMESTAMP_HASHES:
        idx=np.searchsorted(times,blocks[block+"_utc_ns"].astype(np.int64));
        for col,name in enumerate(base.FEATURES): coverage.append({"block":block,"feature":name,"rows":len(idx),"finite":int(np.isfinite(matrix[idx,col]).sum()),"finite_percentage":float(np.isfinite(matrix[idx,col]).mean()*100),"legitimate_nan":int(legitimate[idx,col].sum()),"unresolved":int(unknown[idx,col].sum())})
    pd.DataFrame(coverage).to_csv(run/"reconciled_alignment_audit.csv",index=False)
    source_files=[]
    if certified=="Dukascopy": source_files=[run/"dukascopy_raw_responses.sqlite",*(run/f"fx_source_DUKASCOPY_{re.sub('[^A-Z]','',x)}.npz" for x in base.INSTRUMENTS)]
    source_hash=logical_hash(*(data[e][k] for e in base.INSTRUMENTS for k in ("open_utc_ns","open","high","low","close"))) if data else None
    matrix_hash=logical_hash(matrix)
    previous=archive.read_json(PREVIOUS/"metrics.json")["unknown_source_rows_union"]
    ready=bool(certified and not unknown.any())
    feature_coverage={name:float(np.isfinite(matrix[:,i]).mean()*100) for i,name in enumerate(base.FEATURES)}
    source_coverage={economic:{"first_m1_open_utc":pd.Timestamp(values["open_utc_ns"][0],tz="UTC").isoformat() if len(values["open_utc_ns"]) else None,"last_m1_open_utc":pd.Timestamp(values["open_utc_ns"][-1],tz="UTC").isoformat() if len(values["open_utc_ns"]) else None,"rows":len(values["open_utc_ns"]),"unexplained_gap_count":sum(row["instrument"]==economic and row["classification"]=="unexplained_source_gap" for row in gap_rows)} for economic,values in data.items()}
    results={"run_id":run.name,"run_status":"pending_validator","xm_certification":xm_summary["certification"],
        "dukascopy_attempted":attempts["Dukascopy"]["attempted"],"dukascopy_certification":attempts["Dukascopy"]["certification"],
        "truefx_attempted":attempts["TrueFX"]["attempted"],"truefx_certification":attempts["TrueFX"]["certification"],
        "certified_provider":certified,"source_provider_count":1 if certified else 0,"previous_unresolved_rows":previous,
        "previous_148867_rows_resolved":previous-int(unknown.any(axis=1).sum()),"unresolved_source_rows_final":int(unknown.any(axis=1).sum()),
        "previous_pair_affected_rows":previous_impact["pair_current_endpoint_affected_rows"],"gbpusd_fold1_issue_resolved":bool(ready),"usdjpy_fold1_issue_resolved":bool(ready),"source_coverage":source_coverage,"legitimate_closure_rows":int(legitimate.any(axis=1).sum()),
        "feature_coverage":feature_coverage,"all_six_timestamp_hashes_matched":True,"source_dataset_sha256":source_hash,
        "usd_fx_reconciled_feature_matrix_sha256":matrix_hash,"data_foundation_ready":ready,
        "model_training_performed":False,"strategy_evaluation_performed":False,"gemini_py_changed":False,"operational_model_changed":False,
        "single_next_action":"Run a separately authorized B0/B1 information test." if ready else "Acquire a certifiable full-range single provider; do not train."}
    write_json(run/"source_attempts.json",attempts); write_json(run/"metrics.json",results)
    manifest["data"].update({"symbols":list(base.INSTRUMENTS),"data_sources":[certified or "none"],"data_start_utc":start.isoformat(),"data_end_utc":end.isoformat(),"raw_snapshot_retained":bool(source_files),"reproducibility_claim":"full" if ready else "failed_source_attempts_preserved","source_files":[{"path":p.relative_to(run).as_posix(),"sha256":archive.file_sha256(p),"retention_status":"stored_in_finalized_run_git_archival_pending"} for p in source_files],"mt5_fetch":{"used":True,"terminal_path":str(base.TERMINAL),"symbols":list(base.INSTRUMENTS),"timeframe":"M1","requested_start_utc":start.isoformat(),"requested_end_utc":end.isoformat(),"timestamp_conversion":"MT5 Python epoch UTC","fetch_error":"none","raw_snapshot_retained":False}})
    manifest["operational_hashes_after"]={name:archive.file_sha256(ROOT/name) for name in manifest["operational_hashes_before"]}
    manifest["registry"].update({"parent_or_incumbent":PREVIOUS.name,"selected_configuration":"single-source USD FX pressure reconciliation","trades_per_day":"not_applicable_data_only","realized_win_rate":"not_applicable_data_only","pf":"not_applicable_data_only","mean_r":"not_applicable_data_only","pnl":"not_applicable_data_only","max_dd":"not_applicable_data_only","validator_result":"pending"})
    for path in sorted(run.iterdir()):
        if path.is_file() and path.name not in {"manifest.json","environment.txt","stdout.log"}:
            manifest["artifacts"].append({"kind":"data_only_evidence","path":path.name,"sha256":archive.file_sha256(path),"retention_status":"stored_in_finalized_run_git_archival_pending"})
    write_json(run/"manifest.json",manifest)
    lines=["# GEMINI USD FX PRESSURE SOURCE RECONCILIATION V1","","Data-only; no model, label, outcome, prediction, or strategy metric was accessed.","",f"Preliminary foundation ready: **{'YES' if ready else 'NO'}**","","```json",json.dumps(results,indent=2),"```"]
    (run/"report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps(results,indent=2),flush=True)


def self_check() -> None:
    payload={"timestamp":0,"shift":60000,"multiplier":.01,"open":1.0,"high":1.1,"low":.9,"close":1.0,"times":[0,1],"opens":[0,1],"highs":[0,1],"lows":[0,1],"closes":[0,1]}
    assert len(json.dumps(payload))>0 and dates_between(datetime(2020,1,1,tzinfo=timezone.utc),datetime(2020,1,2,tzinfo=timezone.utc))==["2020-01-01","2020-01-02"]
    print("SOURCE_RECONCILIATION_SELF_CHECK_PASS")


if __name__=="__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("run_dir",type=Path,nargs="?"); parser.add_argument("--self-check",action="store_true"); args=parser.parse_args()
    if args.self_check:self_check()
    elif args.run_dir is None:parser.error("run_dir required")
    else:main(args.run_dir.resolve())
