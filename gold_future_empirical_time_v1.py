"""Deterministic empirical time checks; no recurrence or broker-internal claim."""
import math
from datetime import datetime, timezone

IDENTITY = {'source_id': 'XMGlobal-MT5-6_GOLD', 'symbol': 'GOLD#',
            'broker': 'XM Global Limited', 'server': 'XMGlobal-MT5 6', 'environment': 'demo'}
ALLOWED = (7200, 10800)
SKEW = 5


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def identity(value):
    require(all(value.get(k) == v for k, v in IDENTITY.items()), 'Source identity conflict')


def infer(raw_epoch, observed_utc):
    """Nearest whole-hour candidate derived before testing the frozen allowlist."""
    require(type(raw_epoch) is int and type(observed_utc) in (int, float)
            and math.isfinite(observed_utc), 'Finite scalar timestamps required')
    delta = raw_epoch - observed_utc
    candidate = math.floor(delta / 3600 + 0.5) * 3600
    error = delta - candidate
    return {'observed_offset_seconds': candidate, 'direct_utc_offset_seconds': delta,
            'normalized_utc_error_seconds': error, 'allowed': candidate in ALLOWED,
            'plausible': abs(error) <= SKEW}


def normalize(raw_epoch, offset):
    require(type(raw_epoch) is int and type(offset) is int and offset in ALLOWED, 'Unapproved offset')
    return datetime.fromtimestamp(raw_epoch - offset, timezone.utc).isoformat()


def classify_segment(segment):
    identity(segment['identity'])
    epochs = segment['raw_epochs']
    require(all(type(v) is int for v in epochs), 'Epoch integers')
    monotonic = all(b > a for a, b in zip(epochs, epochs[1:]))
    spacing = all(v % 60 == 0 for v in epochs) and all((b-a) % 60 == 0 for a, b in zip(epochs, epochs[1:]))
    anchors = segment['anchors']
    estimates = [infer(a['raw_epoch'], a['observed_utc_epoch']) for a in anchors]
    stable = len({e['observed_offset_seconds'] for e in estimates}) == 1
    anchor_order = all(b['observed_utc_epoch'] > a['observed_utc_epoch'] and b['raw_epoch'] > a['raw_epoch'] for a, b in zip(anchors, anchors[1:]))
    if not monotonic or not spacing or not anchor_order or any(not e['allowed'] or not e['plausible'] for e in estimates):
        status = 'CONTRADICTED'
    elif anchors and not stable:
        status = 'AMBIGUOUS_TRANSITION'
    elif anchors:
        status = 'OPERATIONAL_UTC_ANCHORED'
    else:
        status = 'EMPIRICAL_INTERNAL_CONSISTENCY' if epochs else 'INSUFFICIENT_DATA'
    return {'classification': status, 'sample_count': len(epochs),
            'raw_epoch_min': min(epochs) if epochs else None, 'raw_epoch_max': max(epochs) if epochs else None,
            'observed_offset_seconds': estimates[0]['observed_offset_seconds'] if estimates and stable else None,
            'direct_utc_offset_summary': [e['direct_utc_offset_seconds'] for e in estimates],
            'normalized_utc_error_summary': [e['normalized_utc_error_seconds'] for e in estimates],
            'monotonic': monotonic, 'bar_spacing_valid': spacing,
            'source_symbol': segment['identity']['symbol'], 'source_server': segment['identity']['server']}


def decision(segments, clock):
    results = [classify_segment(s) for s in segments]
    offsets = sorted({r['observed_offset_seconds'] for r in results
                      if r['classification'] == 'OPERATIONAL_UTC_ANCHORED'})
    contradictions = any(r['classification'] in {'CONTRADICTED', 'AMBIGUOUS_TRANSITION'} for r in results)
    required = {'winter', 'summer', 'march_before', 'march_after', 'october_before', 'october_after', 'current'}
    anchored = {s['regime'] for s, r in zip(segments, results) if r['classification'] == 'OPERATIONAL_UTC_ANCHORED' and len(s['anchors']) >= 2}
    coverage = [{'start': s['anchors'][0]['observed_utc_epoch'], 'end': s['anchors'][-1]['observed_utc_epoch'],
                 'offset': r['observed_offset_seconds']} for s, r in zip(segments, results)
                if r['classification'] == 'OPERATIONAL_UTC_ANCHORED' and len(s['anchors']) >= 2]
    consistent = all(c['start'] < c['end'] for c in coverage)
    ordered = sorted(coverage, key=lambda c: c['start'])
    consistent = consistent and all(a['end'] < b['start'] or a['offset'] == b['offset'] for a, b in zip(ordered, ordered[1:]))
    passed = not contradictions and required <= anchored and offsets == list(ALLOWED) and consistent and clock['quality'] == 'INDEPENDENTLY_VERIFIED'
    status = 'FAIL' if contradictions else 'PASS' if passed else 'PARTIAL'
    return {'status': status, 'policy': 'CONTRADICTED' if contradictions else 'EMPIRICALLY_CERTIFIED_SOURCE_TIME_RULE' if passed else
            'EMPIRICALLY_SUPPORTED_BUT_INSUFFICIENT' if offsets else 'UNRESOLVED',
            'observed_offsets_seconds': offsets, 'allowed_offsets_seconds': list(ALLOWED),
            'empirical_coverage_start': datetime.fromtimestamp(min(c['start'] for c in coverage), timezone.utc).isoformat() if passed else None,
            'empirical_coverage_end': datetime.fromtimestamp(max(c['end'] for c in coverage), timezone.utc).isoformat() if passed else None,
            'certified_segments': coverage if passed else [], 'unobserved_intervals_excluded': True,
            'official_attestation_required': False, 'broker_internal_semantics_claimed': False,
            'runtime_fail_closed': True, 'clock_quality': clock['quality'],
            'blockers': [] if passed else ['Historical seasonal/transition samples lack independent offset anchors',
                         'No certified temporal segment coverage; do not interpolate or invent recurrence'] +
                        ([] if clock['quality'] == 'INDEPENDENTLY_VERIFIED' else ['System UTC is an operational anchor only'])}


def runtime_validate(payload, previous=None):
    """Check one closed native snapshot; transitions require fresh recertification."""
    identity(payload['identity'])
    require(payload['clock_quality'] == 'INDEPENDENTLY_VERIFIED', 'Clock quality insufficient')
    require(payload['timeframe'] in payload['native_timeframes'], 'Native timeframe required')
    require(payload['native_source'] is True and payload['resampled'] is False, 'Native source required')
    anchors = payload['anchors']
    require(len(anchors) >= 2, 'Bracketing anchors required')
    estimates = [infer(a['raw_epoch'], a['observed_utc_epoch']) for a in anchors]
    require(all(e['allowed'] and e['plausible'] for e in estimates), 'Runtime offset/skew conflict')
    offsets = {e['observed_offset_seconds'] for e in estimates}
    require(len(offsets) == 1, 'Ambiguous transition; recertification required')
    require(all(b['observed_utc_epoch'] > a['observed_utc_epoch'] and b['raw_epoch'] > a['raw_epoch']
                for a, b in zip(anchors, anchors[1:])), 'Anchor clock reversal')
    offset = offsets.pop()
    require(payload['certified_offset_seconds'] == offset, 'New regime needs recertification')
    require(payload['regime_certification_id'], 'Regime certificate required')
    epochs = payload['raw_epochs']; successors = payload['successor_raw_epochs']
    require(epochs and len(epochs) == len(successors), 'Closed-bar successor proof required')
    require(all(type(t) is int and t % 60 == 0 for t in epochs+successors), 'Bar-grid conflict')
    require(all(b > a for a, b in zip(epochs, epochs[1:])), 'Raw timestamp reversal')
    require(all(b > a for a, b in zip(epochs, successors)), 'Bar not closed')
    require(successors[:-1] == epochs[1:], 'Successor linkage')
    frame = payload['timeframe']
    nominal = int(frame[1:]) * (60 if frame[0] == 'M' else 3600) if frame[1:].isdigit() else {'Daily': 86400, 'Weekly': 604800, 'Monthly': None}[frame]
    if nominal:
        require(all((b-a) % nominal == 0 for a, b in zip(epochs, successors)), 'Unexpected native spacing')
    else:
        require(all(datetime.fromtimestamp(t, timezone.utc).day == 1 for t in epochs+successors), 'Monthly boundary conflict')
    require(all(anchors[0]['observed_utc_epoch'] <= t-offset < close-offset <= anchors[-1]['observed_utc_epoch']
                for t, close in zip(epochs, successors)), 'Bar outside observed stable regime')
    if previous:
        require(epochs[0]-offset > previous['last_normalized_epoch'], 'Backward normalized timestamp')
        require(offset == previous['offset'] or
                (payload['regime_certification_id'] != previous['regime_certification_id']
                 and payload['transition_recertified'] is True), 'Offset change requires recertification')
    return {'status': 'PASS', 'offset': offset, 'raw_epochs': list(epochs),
            'normalized_timestamps': [normalize(t, offset) for t in epochs],
            'last_normalized_epoch': epochs[-1]-offset,
            'regime_certification_id': payload['regime_certification_id'],
            'gap_policy': 'Actual source bars only; no synthesis or forward-fill'}
