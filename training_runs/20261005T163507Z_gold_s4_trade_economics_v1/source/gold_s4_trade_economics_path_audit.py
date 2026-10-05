"""Pure exit-path functions. Historical input is loaded only by the USER runner."""
import math
from datetime import datetime, timezone, timedelta


def replay_entries(entries, bars, rule):
    """Stop the entire candidate at first conflict; never skip or move an entry."""
    ledger = []
    balance, last_loss = 1000., None
    rewards = []
    for entry in entries:
        now = entry['entry_epoch']
        if ledger:
            previous = ledger[-1]
            open_exit = previous['exit_reason'] in ('timeout', 'no_progress')
            if previous['exit_epoch'] > now or (previous['exit_epoch'] == now and not open_exit):
                return ledger, 'REFERENCE_ENTRY_OVERLAP', entry['trade_id']
        if last_loss is not None and now - last_loss < 15 * 60:
            return ledger, 'REFERENCE_COOLDOWN_CONFLICT', entry['trade_id']
        local_date = datetime.fromtimestamp(now, timezone.utc).astimezone(timezone(timedelta(hours=8))).date()
        daily = sum(t['account_pnl'] for t in ledger if datetime.fromtimestamp(
            t['exit_epoch'], timezone.utc).astimezone(timezone(timedelta(hours=8))).date() == local_date)
        if daily <= -balance * .05:
            return ledger, 'REFERENCE_DAILY_LOSS_CONFLICT', entry['trade_id']
        risk_mult = 1.
        recent = rewards[-30:]
        if len(recent) >= 18:
            gp, gl = sum(v for v in recent if v > 0), -sum(v for v in recent if v <= 0)
            if gl > 0 and gp/gl < 1.15:
                risk_mult *= .5
        result = replay_trade(entry, bars[entry['entry_index']:], rule)
        result['risk_mult'] = risk_mult
        result['balance_before'] = balance
        result['risk_budget'] = max(balance, 0.) * .014 * risk_mult
        result['account_pnl'] = result['net_r'] * result['risk_budget']
        balance += result['account_pnl']
        result['balance_after'] = balance
        rewards.append(result['account_pnl'])
        if result['net_r'] <= 0:
            last_loss = result['exit_epoch']
        ledger.append(result)
    return ledger, None, None


def replay_trade(entry, bars, rule):
    """Bars are (UTC epoch seconds, open, high, low, close); long trades only.

    R unit is the ORIGINAL entry stop distance plus original spread. Changing
    stops never rescales the denominator or position size. Amendments to a stop
    become active on the NEXT available bar, after old-stop/target processing.
    """
    price = entry['entry_price']
    unit = entry['sl_distance']
    spread = entry['spread_points']
    if not all(math.isfinite(v) for v in (price, unit, spread)) or unit <= 0 or spread < 0:
        raise ValueError('Invalid frozen entry risk/cost')
    stop = price - unit * rule['stop_multiple']
    target = price + (entry['tp_distance'] if rule['target_r'] is None else unit * rule['target_r'])
    stop_kind = 'stop_loss'
    mfe = mae = 0.0
    start = entry['entry_epoch']
    previous = None
    for offset, bar in enumerate(bars):
        timestamp, opening, high, low, close = bar
        if (not all(math.isfinite(v) for v in bar) or low > min(opening, close)
                or high < max(opening, close) or low > high
                or timestamp < start or previous is not None and timestamp <= previous):
            raise ValueError('Invalid causal OHLC path')
        if offset == 0 and (timestamp != start or opening != price):
            raise ValueError('Frozen entry timestamp/price mismatch')
        previous = timestamp
        elapsed = (timestamp - start) / 60
        reason = None
        both = False
        # OPEN-only decisions use completed preceding bars, never this bar HIGH/LOW.
        if elapsed >= rule['time_minutes']:
            fill, reason = opening, 'timeout'
        elif rule['no_progress_minutes'] is not None and elapsed >= rule['no_progress_minutes'] and mfe < .2 * unit:
            fill, reason = opening, 'no_progress'
        else:
            hit_stop, hit_target = low <= stop, high >= target
            both = hit_stop and hit_target
            if hit_stop:
                fill, reason = stop, stop_kind
            elif hit_target:
                fill, reason = target, 'take_profit'
        if reason:
            # Exit-bar extremes are unordered: do not report unknowable post-exit MFE/MAE.
            if reason in ('timeout', 'no_progress'):
                mfe = max(mfe, fill - price)
                mae = max(mae, price - fill)
            return _fill(entry, timestamp, fill, reason, both, mfe, mae, offset)
        mfe = max(mfe, high - price)
        mae = max(mae, price - low)
        if rule['break_even_r'] is not None and mfe >= unit * rule['break_even_r'] and stop < price:
            stop, stop_kind = price, 'break_even'
        if rule['trail_after_r'] is not None and mfe >= unit * rule['trail_after_r']:
            proposed = price + mfe - unit * rule['trail_by_r']
            if proposed > stop:
                stop, stop_kind = proposed, 'trailing'
    if previous is None:
        raise ValueError('Empty trade path')
    # Exact legacy cohort-end fallback; NEVER fetch beyond the approved universe.
    return _fill(entry, previous, close, 'cohort_end', False, mfe, mae, offset)


def _fill(entry, timestamp, price, reason, both, mfe, mae, offset):
    denominator = entry['sl_distance'] + entry['spread_points'] * .01
    gross = price - entry['entry_price']
    return dict(entry, exit_epoch=timestamp, exit_price=price, exit_reason=reason,
                same_bar_both_hit=bool(both), gross_price=gross, gross_r=gross / denominator,
                net_r=(gross - (entry['spread_points'] + 5) * .01) / denominator,
                stress_r=(gross - (entry['spread_points'] + 10) * .01) / denominator,
                holding_minutes=(timestamp - entry['entry_epoch']) / 60,
                mfe_completed_bar_r=mfe / denominator, mae_completed_bar_r=mae / denominator,
                bars_to_exit=offset)


def verify_audit(root):
    import ast
    import hashlib
    import json
    audit = json.loads((root/'gold_s4_trade_economics_path_audit.json').read_text(encoding='utf-8'))
    path = root/audit['baseline_source']
    if hashlib.sha256(path.read_bytes()).hexdigest() != audit['baseline_source_sha256']:
        raise ValueError('Baseline execution source changed')
    assignments = {node.targets[0].id: node.value for node in ast.parse(path.read_text(encoding='utf-8')).body
                   if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)}
    for name, expected in audit['baseline_constants'].items():
        if ast.literal_eval(assignments[name]) != expected:
            raise ValueError('Baseline semantics changed: ' + name)
    return audit
