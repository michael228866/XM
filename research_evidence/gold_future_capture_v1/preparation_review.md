# GOLD capture preparation and collector review

Status: BLOCKED at Phase 1 source certification. Formal certification may record this failure; dependent implementation/activation is deferred under the explicit stop rule.

User-verified metadata: connected=true, build=6182, XM Global Limited / XMGlobal-MT5 6, GOLD#, description GOLD, path Derivatives\Spot Metals Ultra Low\GOLD#, digits=2, point=0.01, visible=true. These are user statements, not independently sampled terminal evidence. No MT5 connection or account query occurred.

Source attestation remains template=true and environment=unknown. Timezone remains UNRESOLVED. See hashed public evidence notes: generic MT5 UTC documentation does not reconcile the legacy conversion or establish the source-specific calendar. No approved native 21-timeframe continuity/schema/freshness evidence or exact continuous certified EMA/MACD prefix exists in these supplied inputs. API timeframe constants alone do not certify source availability. No resampling chosen.

## Static review of existing collector

- Lines 98-112 append to daily CSV with mode a; prior files remain mutable. Flush occurs, fsync does not.
- Lines 77-83 replace mutable JSON through a temporary file. This is not immutable sealed snapshot publication.
- Lines 160-172 preserve raw epoch, OHLC and spread but apply the unauthoritative EET/EEST helper imported at lines 13-17.
- Lines 385-486 use mutable progress state and replace the final manifest. No linked predecessor manifest verification, sealed snapshot sequence, orphan quarantine or immutable revision snapshots exists.
- Collection covers M1/ticks, not the required native M1 plus 20 higher timeframes. No certified source/timezone attestation binding or bound dependency inventory.
- No direct strategy/model call or production file write was observed in this source review. This is not an operational or transitive dependency certification.

COLLECTOR_STATIC_STATUS=NOT_STATICALLY_CONFORMANT. A new collector is needed eventually, but source certification stopped before that dependent implementation phase. Old collector unchanged and never executed.

No protocol freeze, capture root, activation, holdout start or interim strategy metrics. Historical discovery INTERESTING and confirmation SUPPORTIVE remain development context only. No production change or promotion.

Formal wrapper preserves clean pre-create Git status and all committed input bytes. Creating a tracked run directory can make the certification report repo_dirty=true; the validator must independently prove that every changed path is confined to that newly created run and that tracked source bytes match the pushed source commit. This does not claim the report itself observed a clean worktree.

The original certifier rejects output under training_runs. Its unchanged guard is retained: output goes to a unique ignored gold_*.json at repository root, then an identical copy is retained in the formal archive; neither copy is deleted or overwritten.
