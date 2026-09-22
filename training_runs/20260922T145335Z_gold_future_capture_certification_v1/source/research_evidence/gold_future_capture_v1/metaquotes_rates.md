# Public documentation review

Source URL: https://www.mql5.com/en/docs/python_metatrader5/mt5copyratesfrom_py
Retrieved at UTC: 2026-09-22T14:48:49.082674+00:00
Authority: FEED_DOCUMENT
Reviewed by: Codex (document review only)
Reviewed at UTC: 2026-09-22T14:48:49.082674+00:00

Official API documentation describes bar open timestamps as UTC without shift and enumerates the native timeframe API. This does not attest XMGlobal-MT5 6 GOLD# actual timestamp encoding, DST transitions, session calendar, native timeframe history continuity or prefix identity. Existing collector applies an EET/EEST conversion; the discrepancy requires source-specific adjudication.
