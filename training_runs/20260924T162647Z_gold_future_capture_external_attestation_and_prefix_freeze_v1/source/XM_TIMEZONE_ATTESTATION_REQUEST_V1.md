# XM source/timezone attestation request — draft, NOT SENT

To: XM Global Limited support
Subject: Written timestamp and trading-calendar confirmation for XMGlobal-MT5 6 / GOLD#

Please confirm the following specifically for **XMGlobal-MT5 6**, demo
environment, exact symbol **GOLD#** (2 digits, point 0.01). Please distinguish
this source from other XM entities, servers and GOLD symbols. No credentials,
account identifiers or financial account details are included in this request.

1. What is the server display-clock timezone? Specify winter and summer UTC
   offsets, their effective dates, and any historical changes.
2. What exact DST transition rule applies (date algorithm, transition time and
   timezone basis)? Is it Europe/Athens, EET/EEST, US DST, or another rule?
   Please provide authoritative effective coverage and future recurrence policy.
3. For GOLD#, what are the daily session open/close, daily rollover, Friday
   close and Sunday/Monday reopening times? Specify timezone and DST exceptions.
4. Where are authoritative holiday-hours updates published for this exact
   server/instrument? Is terminal internal mail authoritative? Provide the
   official publication URL/channel and revision/effective-time policy.
5. Do native M1/M2/M3/M4/M5/M6/M10/M12/M15/M20/M30/H1/H2/H3/H4/H6/H8/H12/
   Daily/Weekly/Monthly timestamps identify BAR_OPEN? How are weekly/monthly
   opens, closures and missing sessions defined?
6. Do Python MetaTrader5 copy_rates_* returned epoch integers encode true Unix
   UTC or server-local wall time without its offset? Is the rule the same for
   symbol_info_tick and historical bars across all native timeframes?
7. Please explain how to reconcile the MetaQuotes UTC API documentation with
   retained current observations on the configured terminal where the raw
   tick epoch is approximately 10,799 seconds ahead of system UTC and a
   hypothesized 3-hour correction aligns it within one second.
8. Please provide at least one independently anchored historical transition
   or session timestamp, preferably winter, summer, March and October examples.

Please identify the responding XM entity, document/version, applicable server,
instrument, validity range, and whether the response authorizes future rule
extrapolation. Partial answers are welcome; unanswered fields remain unresolved.

This request concerns raw timestamp/source integrity only. It does not ask for
trading advice, account activity, model outputs or performance information.

Delivery status: NOT_SENT. No authorized outbound communication channel is
available in this session. A future response must be retained, timestamped,
hashed and independently reviewed before it can support certification.
