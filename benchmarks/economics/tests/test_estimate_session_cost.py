import importlib.util
from datetime import datetime, timedelta, timezone
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('cost', Path(__file__).parents[1] / 'estimate_session_cost.py')
cost = importlib.util.module_from_spec(spec); spec.loader.exec_module(cost)

class CostBudgetTest(unittest.TestCase):
    def setUp(self):
        self.now = datetime.now(timezone.utc)
        self.config = {'currency': 'USD', 'prices_checked_at': self.now.isoformat(),
            'price_source': 'https://cloud.google.com/compute/all-pricing',
            'minutes': dict(zip(cost.PHASES, [5, 10, 15, 20, 10])),
            'compute_usd_per_hour': 3, 'disk_gb': 100, 'disk_usd_per_gb_hour': .001,
            'transfer_gb': 1, 'transfer_usd_per_gb': .1,
            'spent_usd': 0, 'pending_usd': 0, 'session_cap_usd': 8}

    def test_all_phases_storage_transfer_count_toward_budget(self):
        result = cost.estimate(self.config, self.now)
        self.assertAlmostEqual(result['estimated_usd'], 3.2)
        self.assertEqual(result['deadline_utc'], (self.now + timedelta(hours=1)).isoformat())

    def test_reserve_and_pending_charges_cannot_be_spent(self):
        self.config.update(spent_usd=35, pending_usd=2)
        with self.assertRaises(ValueError): cost.estimate(self.config, self.now)

    def test_stale_prices_and_unofficial_sources_fail(self):
        self.config['prices_checked_at'] = (self.now - timedelta(days=2)).isoformat()
        with self.assertRaises(ValueError): cost.estimate(self.config, self.now)
        self.config['prices_checked_at'] = self.now.isoformat()
        self.config['price_source'] = 'https://example.com/price'
        with self.assertRaises(ValueError): cost.estimate(self.config, self.now)

    def test_nan_and_missing_download_phase_fail(self):
        self.config['compute_usd_per_hour'] = float('nan')
        with self.assertRaises(ValueError): cost.estimate(self.config, self.now)
        self.config['compute_usd_per_hour'] = 3
        del self.config['minutes']['download']
        with self.assertRaises(KeyError): cost.estimate(self.config, self.now)

    def test_zero_price_placeholders_do_not_pass(self):
        self.config['compute_usd_per_hour'] = 0
        with self.assertRaises(ValueError): cost.estimate(self.config, self.now)
