"""Bounded actual-feed timestamp observations; never inspect price values."""
import argparse
import json
import locale
import subprocess
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from gold_mt5_timestamp_semantics_diagnostic_v1 import session
from gold_future_empirical_time_v1 import IDENTITY, classify_segment, identity

WINDOWS = [('winter', '2026-01-15'), ('summer', '2026-07-15'),
           ('march_before', '2026-03-27'), ('march_after', '2026-03-30'),
           ('october_before', '2025-10-24'), ('october_after', '2025-10-27')]


def diagnose():
    clock = subprocess.run(['w32tm', '/query', '/status'], capture_output=True, timeout=15, check=False)
    clock_doc = {'system_clock_source': 'OS UTC; Windows Time query retained as an operational check',
                 'system_clock_check': {'exit_code': clock.returncode,
                    'stdout': clock.stdout.decode(locale.getpreferredencoding(False), errors='replace'),
                    'stderr': clock.stderr.decode(locale.getpreferredencoding(False), errors='replace')},
                 'max_allowed_clock_skew_seconds': 5, 'quality': 'OPERATIONAL_CHECK_ONLY',
                 'limitation': 'Service status alone does not independently bound UTC clock skew'}
    segments = []
    result = {'segments': segments, 'clock': clock_doc, 'identity_verified': False,
              'price_values_inspected': False, 'generated_at_utc': datetime.now(timezone.utc).isoformat()}
    try:
        with session() as (mt5, metadata):
            current_identity = {'source_id': IDENTITY['source_id'], 'symbol': metadata['symbol'],
                                'broker': metadata['broker_company'], 'server': metadata['broker_server'],
                                'environment': metadata['source_account_environment']}
            identity(current_identity)
            result.update(identity_verified=True, identity=current_identity)
            for regime, date in WINDOWS:
                start = datetime.fromisoformat(date+'T12:00:00+00:00'); end = start+timedelta(minutes=5)
                rates = mt5.copy_rates_range('GOLD#', mt5.TIMEFRAME_M1, start, end)
                epochs = [] if rates is None else [int(r['time']) for r in rates]
                del rates
                if len(epochs) > 6:
                    raise ValueError('Historical sample bound')
                seg = {'segment_id': regime, 'regime': regime, 'sample_start': start.isoformat(), 'sample_end': end.isoformat(),
                       'raw_epochs': epochs, 'anchors': [], 'identity': current_identity,
                       'anchor_scope': 'No independent historical UTC anchor; no inferred absolute offset'}
                seg.update(classify_segment(seg)); segments.append(seg)
            for index in range(3):
                anchors = []
                for sample in range(2):
                    tick = mt5.symbol_info_tick('GOLD#')
                    if tick is None:
                        raise ValueError('Current tick unavailable')
                    anchors.append({'raw_epoch': int(tick.time), 'observed_utc_epoch': datetime.now(timezone.utc).timestamp()})
                    del tick
                    if sample == 0:
                        time.sleep(2)
                rates = mt5.copy_rates_from_pos('GOLD#', mt5.TIMEFRAME_M1, 1, 3)
                epochs = [] if rates is None else [int(r['time']) for r in rates]
                del rates
                seg = {'segment_id': 'current_'+str(index), 'regime': 'current',
                       'sample_start': datetime.fromtimestamp(anchors[0]['observed_utc_epoch'], timezone.utc).isoformat(),
                       'sample_end': datetime.fromtimestamp(anchors[-1]['observed_utc_epoch'], timezone.utc).isoformat(),
                       'raw_epochs': epochs, 'anchors': anchors, 'identity': current_identity,
                       'anchor_scope': 'Operational OS UTC; not sole external certification evidence'}
                seg.update(classify_segment(seg)); segments.append(seg)
    except Exception as error:
        result['error_type'] = type(error).__name__
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or (args.output.parent/'FINALIZED.json').exists():
        raise FileExistsError('Immutable diagnostic output')
    result = diagnose()
    with args.output.open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2); f.write('\n')
    print('EMPIRICAL_SEGMENTS='+str(len(result['segments'])))
