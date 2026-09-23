# Reviewed official documentation

Source URL: https://www.mql5.com/en/docs/constants/environment_state/accountinformation
Retrieved at UTC: 2026-09-23T06:16:31.128635+00:00
Authority: FEED_DOCUMENT
Reviewed by: Codex (document review; not broker attestation)
Reviewed at UTC: 2026-09-23T06:16:31.128635+00:00

Supported claims:
- ACCOUNT_TRADE_MODE distinguishes demo, contest and real environments.
- Only company, server and environment classification are extracted in memory; full account structures are never persisted.

Limitations:
- Contest is not relabeled demo or live. A failed query remains unknown.
