"""Minimal MT5 identity extraction authorized by the v4 task addendum."""
from contextlib import contextmanager
from pathlib import Path

TERMINAL = r'D:\XM2\terminal64.exe'
EXPECTED = {
    'broker_company': 'XM Global Limited',
    'broker_server': 'XMGlobal-MT5 6',
    'source_account_environment': 'demo',
}


def extract_identity(mt5):
    """Never serialize the account object or retain unrelated account fields."""
    try:
        account = mt5.account_info()
    except Exception:
        raise ValueError('SOURCE_IDENTITY_ACCOUNT_INFO_UNAVAILABLE') from None
    if account is None:
        raise ValueError('SOURCE_IDENTITY_ACCOUNT_INFO_UNAVAILABLE')
    try:
        company = getattr(account, 'company', None)
        server = getattr(account, 'server', None)
        mode = getattr(account, 'trade_mode', None)
    finally:
        del account
    matches = [label for name, label in (
        ('ACCOUNT_TRADE_MODE_DEMO', 'demo'),
        ('ACCOUNT_TRADE_MODE_REAL', 'live'),
        ('ACCOUNT_TRADE_MODE_CONTEST', 'contest'),
    ) if type(mode) is int and type(getattr(mt5, name, None)) is int
        and mode == getattr(mt5, name)]
    environment = matches[0] if len(matches) == 1 else 'unknown'
    # Strings only; never allow an unexpected object to reach serialization.
    return {
        'broker_company': company if type(company) is str else None,
        'broker_server': server if type(server) is str else None,
        'source_account_environment': environment,
    }


def verify_identity(mt5):
    observed = extract_identity(mt5)
    for key, expected in EXPECTED.items():
        if observed[key] != expected:
            raise ValueError('SOURCE_IDENTITY_MISMATCH:' + key)
    terminal = mt5.terminal_info()
    symbol = mt5.symbol_info('GOLD#')
    if terminal is None or not terminal.connected:
        raise ValueError('SOURCE_TERMINAL_DISCONNECTED')
    if Path(terminal.path).resolve() != Path(TERMINAL).parent.resolve():
        raise ValueError('SOURCE_TERMINAL_PATH_MISMATCH')
    if symbol is None or symbol.name != 'GOLD#':
        raise ValueError('SOURCE_SYMBOL_MISMATCH')
    if symbol.digits != 2 or symbol.point != 0.01:
        raise ValueError('SOURCE_SYMBOL_PRECISION_MISMATCH')
    return {
        **observed,
        'source_id': 'XMGlobal-MT5-6_GOLD', 'symbol': 'GOLD#',
        'digits': 2, 'point': 0.01, 'terminal_build': int(terminal.build),
        'terminal_path': TERMINAL, 'mt5_package_version': str(mt5.__version__),
        'transport': 'MetaTrader5 Python API',
        'identity_check_method': 'MetaTrader5.account_info sanitized field extraction',
        'identity_check_passed': True,
    }


@contextmanager
def session():
    import MetaTrader5 as mt5
    if not mt5.initialize(TERMINAL, timeout=10000):
        raise ValueError('SOURCE_INITIALIZATION_FAILED')
    try:
        yield mt5, verify_identity(mt5)
    finally:
        mt5.shutdown()
