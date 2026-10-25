#!/usr/bin/env python3
"""Read-only USD estimate; requires fresh, sourced prices and all session phases."""
import argparse
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
from urllib.parse import urlparse

PHASES = ('startup', 'download', 'build', 'benchmark', 'cleanup')


def estimate(config, now=None):
    now = now or datetime.now(timezone.utc)
    if config.get('currency') != 'USD':
        raise ValueError('prices must use USD')
    recorded = datetime.fromisoformat(config['prices_checked_at'].replace('Z', '+00:00'))
    if recorded.tzinfo is None or not timedelta(0) <= now - recorded <= timedelta(hours=24):
        raise ValueError('price verification must be within the last 24 hours')
    source = urlparse(config['price_source'])
    if source.scheme != 'https' or source.hostname not in ('cloud.google.com', 'cloudbilling.googleapis.com'):
        raise ValueError('price_source must link to official GCP pricing or Billing Catalog API')

    def number(key, values=config):
        value = values[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f'{key} must be a finite nonnegative number')
        return value

    minutes = sum(number(phase, config['minutes']) for phase in PHASES)
    if minutes <= 0: raise ValueError('session duration must be positive')
    hourly = number('compute_usd_per_hour')
    disk_gb = number('disk_gb')
    disk_rate = number('disk_usd_per_gb_hour')
    if hourly <= 0 or (disk_gb > 0 and disk_rate <= 0):
        raise ValueError('compute and allocated disk prices must be positive; zero placeholders are not quotes')
    compute = hourly * minutes / 60
    storage = disk_gb * disk_rate * minutes / 60
    transfer = number('transfer_gb') * number('transfer_usd_per_gb')
    total = compute + storage + transfer
    # Reserve is fixed by the approved v0.2 plan. Pending charges remain liabilities.
    spent = number('spent_usd') + number('pending_usd')
    remaining = 50 - 10 - spent
    session_cap = number('session_cap_usd')
    if session_cap > 12: raise ValueError('session cap exceeds the approved per-session allocation')
    if total > session_cap or total > remaining:
        raise ValueError(f'estimated ${total:.2f} exceeds session ${session_cap:.2f} or usable remaining ${remaining:.2f}')
    return {'schema_version': 1, 'estimated_usd': round(total, 6),
        'compute_usd': round(compute, 6), 'disk_usd': round(storage, 6), 'transfer_usd': round(transfer, 6),
        'usable_remaining_after_usd': round(remaining - total, 6), 'reserve_usd': 10,
        'duration_minutes': minutes, 'deadline_utc': (now + timedelta(minutes=minutes)).isoformat(),
        'price_source': config['price_source'], 'prices_checked_at': config['prices_checked_at'],
        'note': 'Estimate only; billing delay, spot interruption and retained resources remain separate liabilities.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    args = parser.parse_args()
    try:
        print(json.dumps(estimate(json.loads(args.config.read_text())), indent=2))
    except (KeyError, ValueError, TypeError) as error:
        parser.exit(1, f'Cost preflight failed: {error}\n')
