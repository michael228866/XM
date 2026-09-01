# DATA FOUNDATION 1 — Historical Information Acquisition Audit

Generated (UTC): `2026-09-01T07:19:25.398384+00:00`

Status: `data_foundation_only`; no model was trained and no strategy outcome was evaluated.

## MT5 historical tick audit

| Symbol | Class | Earliest tick (UTC) | Latest tick (UTC) | Approx ticks | Bid | Ask | Last | Volume | Flags |
| --- | --- | --- | --- | ---: | --- | --- | --- | --- | --- |
| GOLD# | gold | 2023-08-03T22:00:06.401000+00:00 | 2026-09-01T07:19:10+00:00 | 253239402 | yes | yes | no | no | yes |
| SILVER# | silver | 2023-08-03T22:00:01.622000+00:00 | 2026-09-01T07:19:11+00:00 | 154174637 | yes | yes | no | no | yes |
| USDX-SEP26 | usd_index | 2026-06-02T12:59:06.124000+00:00 | 2026-09-01T07:19:04+00:00 | 675316 | yes | yes | no | no | yes |
| VIX-SEP26 | vix | 2026-08-05T10:40:19.524000+00:00 | 2026-09-01T07:11:48+00:00 | 15754 | yes | yes | no | no | yes |
| OILCash# | crude | 2025-07-22T08:55:59.690000+00:00 | 2026-09-01T07:19:09+00:00 | 17096376 | yes | yes | no | no | yes |
| GSOIL-SEP26 | crude | 2026-08-05T10:40:19.524000+00:00 | unavailable | not estimable | no/unverified | no/unverified | no | no | no/unverified |
| OIL-OCT26 | crude | 2026-08-05T10:40:19.524000+00:00 | unavailable | not estimable | no/unverified | no/unverified | no | no | no/unverified |
| OILMn-OCT26 | crude | 2026-08-05T10:40:19.524000+00:00 | unavailable | not estimable | no/unverified | no/unverified | no | no | no/unverified |

Approximate counts are explicitly low-confidence extrapolations from a recent one-hour sample; the script does not download or retain bulk historical ticks.
Month-level availability and every progressive GOLD# boundary query are preserved in the JSON metadata.
GOLD# monthly coverage inside the broker boundary: `2023-08` through `2026-09`; non-available queried months: `none`.

## Microstructure feasibility

GOLD# broker tick boundary: `2023-08-03T22:00:06.401000+00:00`.
Bid/ask reconstruction for the full 2018–2024 interval: `NO`; the available tail begins at the broker boundary in August 2023.
Within the verified tick range, bid/ask-derived spread, changes, tick intensity, signed midquote pressure, realized variance, path efficiency and burst state are feasible only when the JSON field checks are true.

Pre-registered new-data hypothesis: `log1p(TICKVOL).shift(1).diff().diff()` on completed GOLD# M1 bars. Its bar size, transform and lags are frozen; 2018–2024 cannot be reused as independent confirmation.

## Cross-market availability

Timestamp-aligned intraday sources currently verified: MT5 SILVER# (silver), MT5 USDX-SEP26 (usd_index), MT5 VIX-SEP26 (vix), MT5 OILCash# (crude).
Daily/slow context only: U.S. Treasury daily par yields / Federal Reserve DGS2 and DGS10, free official historical VIX daily series when intraday entitlement is absent, official WTI spot/energy series when only daily observations are used.
Contract futures symbols are not continuous series; rolls must be handled without future knowledge before any causal model use.

| Instrument | Source | Granularity | First | Last | Timezone | Missing periods | Revisions | Knowable at GOLD entry? |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| XAGUSD / SILVER# | local MT5 SILVER# M1 export | M1 OHLC + TICKVOL | 2014-01-01T22:00:00+00:00 | 2026-05-28T13:23:00+00:00 | XM terminal EET/EEST server wall time converted to UTC (+2 winter/+3 summer); conversion matches the live tick epoch audit | Not exhaustively gap-audited; no tick path or bid/ask | Frozen local export; upstream broker history can change on re-export | Yes after each bar closes; current/incomplete bar is prohibited |
| SILVER# | XM MT5 broker tick store | quote tick | 2023-08-03T22:00:01.622000+00:00 | 2026-09-01T07:19:11+00:00 | UTC after audited XM EET/EEST conversion | No missing month detected inside audited boundary; weekends/closures expected | Broker history may be backfilled; collector preserves received raw rows | Yes when tick timestamp <= GOLD entry timestamp |
| USDX-SEP26 | XM MT5 broker tick store | quote tick | 2026-06-02T12:59:06.124000+00:00 | 2026-09-01T07:19:04+00:00 | UTC after audited XM EET/EEST conversion | No missing month detected inside audited boundary; weekends/closures expected | Broker history may be backfilled; collector preserves received raw rows | Yes when tick timestamp <= GOLD entry timestamp |
| VIX-SEP26 | XM MT5 broker tick store | quote tick | 2026-08-05T10:40:19.524000+00:00 | 2026-09-01T07:11:48+00:00 | UTC after audited XM EET/EEST conversion | No missing month detected inside audited boundary; weekends/closures expected | Broker history may be backfilled; collector preserves received raw rows | Yes when tick timestamp <= GOLD entry timestamp |
| OILCash# | XM MT5 broker tick store | quote tick | 2025-07-22T08:55:59.690000+00:00 | 2026-09-01T07:19:09+00:00 | UTC after audited XM EET/EEST conversion | No missing month detected inside audited boundary; weekends/closures expected | Broker history may be backfilled; collector preserves received raw rows | Yes when tick timestamp <= GOLD entry timestamp |
| US 2Y constant maturity (DGS2) | Federal Reserve H.15 via FRED: https://fred.stlouisfed.org/series/DGS2 | daily | 1976-06-01 (series start) | 2026-08-28 observed; updated 2026-08-31 15:16 CDT at audit time | observation date plus documented publication timestamp | Weekends, holidays and occasional missing daily observations | FRED states all data are subject to revision | Only after the H.15/FRED publication timestamp, never intraday on observation date |
| US 10Y constant maturity (DGS10) | Federal Reserve H.15 via FRED: https://fred.stlouisfed.org/series/DGS10 | daily | 1962-01-02 (series start) | 2026-08-28 observed; updated 2026-08-31 15:16 CDT at audit time | observation date plus documented publication timestamp | Weekends, holidays and occasional missing daily observations | FRED states all data are subject to revision | Only after the H.15/FRED publication timestamp, never intraday on observation date |
| VIX index | Cboe official daily history: https://www.cboe.com/tradable_products/vix/vix_historical_data/ | daily close in free official file | 1990 | present, updated daily | U.S. market date; exact publication timestamp must be retained on acquisition | Non-trading days; intraday history not included in free file | Methodology/history can be corrected; preserve acquisition vintage | Only after the daily close/publication, not as same-day intraday input |
| WTI Cushing spot | EIA official history: https://www.eia.gov/dnav/pet/hist/rwtca.htm | daily observations published with delay | 1986 | latest published EIA release vintage | observation date plus EIA release timestamp | Non-business days and publication lag | Historical values/source methodology can be revised; preserve vintage | Not knowable intraday on the observation date; use only after release |
| ICE U.S. Dollar Index futures | ICE contract reference: https://www.ice.com/products/194/US-Dollar-Index-Futures/expiry | licensed tick/intraday; contract reference is public | Not acquired as a continuous historical dataset | Current contract-dependent | exchange timestamp; normalize to UTC | Historical intraday feed not locally licensed/acquired | Trades are immutable; vendor corrections and roll construction require vintages | Yes only with a timestamped licensed live/historical feed |

## Economic-event foundation

Official agency schedules can support documented release timestamps for BLS CPI/Employment, BEA GDP/PCE and Federal Reserve FOMC decisions/statements. There is no locally verified unified historical surprise feed. Actual-minus-forecast surprise is therefore excluded.

| Events | Official source | Timestamp basis | Coverage | Acquisition status |
| --- | --- | --- | --- | --- |
| CPI, Core CPI, Employment Situation, NFP, PPI, Initial Claims | [U.S. BLS release schedules and archived releases](https://www.bls.gov/schedule/news_release/bls.ics) | Scheduled date/time is official and stated in Eastern Time; archived release embargo header documents actual release timestamp | Archived Employment releases are listed back to at least 1994; CPI has equivalent archives | Source verified; direct automated ICS request was denied by BLS bot policy in this environment, so validated provenance import is used |
| GDP, PCE, Core PCE, Personal Income and Outlays | [U.S. BEA full release schedule and archived news releases](https://www.bea.gov/news/schedule/full) | Official schedule includes date and Eastern release time; archived news release records the actual embargo time | Current full-year schedule plus archived release documents; revisions/rescheduling must use the actual archive | Source verified; historical canonical extraction not yet materialized |
| FOMC rate decision, FOMC statement | [Federal Reserve FOMC calendars, statements and historical materials](https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm) | Meeting dates and statement documents are official; exact release time must come from the statement/document record rather than an assumed universal time | Current calendar 2021-2027 plus official year-by-year historical materials | Source verified; exact timestamp extraction requires per-document provenance |

## Untouched-forward protocol

Cutoff remains `2026-09-01T02:00:00+00:00`. Raw collection is isolated under `untouched_forward/generation21/data/` and contains no labels, returns, predictions, positions or strategy outcomes.
Collector status: `untouched_raw_only`; contamination: `untouched`; latest incremental run: GOLD#: +615 ticks/+2 M1, SILVER#: +284 ticks/+2 M1, USDX-SEP26: +7 ticks/+2 M1, VIX-SEP26: +0 ticks/+0 M1, OILCash#: +93 ticks/+2 M1, +0 event rows; total stored official event timestamps: 16.

## Required answers

1. Actual GOLD# tick history reaches back to `2023-08-03T22:00:06.401000+00:00`; see progressive probes for timeout/empty evidence.
2. Bid/ask microstructure for all of 2018–2024: `not reconstructable`. Only 2023-08-03 onward is present, so 2018–2022 and most of 2023 have no broker tick-level bid/ask history.
3. Tick-volume acceleration can be independently tested only on untouched bars at or after the cutoff, using the frozen definition above. It cannot receive an independent historical test from already inspected 2018–2024 data.
4. Verified intraday cross-market data: MT5 SILVER# (silver), MT5 USDX-SEP26 (usd_index), MT5 VIX-SEP26 (vix), MT5 OILCash# (crude).
5. Daily-only datasets unsuitable as intraday causal features: U.S. Treasury daily par yields / Federal Reserve DGS2 and DGS10, free official historical VIX daily series when intraday entitlement is absent, official WTI spot/energy series when only daily observations are used.
6. Reliable release timestamps can be assembled from official BLS, BEA and Federal Reserve release/meeting records; forecast surprises remain unavailable until a timestamped, licensed/reproducible source is added.
7. Collect GOLD# ticks and closed M1 bars, observed bid/ask spread, M1 tick count, SILVER# aligned ticks/M1, verified cross-market quotes, and official event timestamps—raw and append-only, without outcomes.
8. Generation 22: `NOT JUSTIFIED` — No complete, long, timestamp-aligned new intraday information family is currently frozen and validated. Continue acquisition; do not train a candidate yet.
