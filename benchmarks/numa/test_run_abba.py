import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('numa_abba', Path(__file__).with_name('run_abba.py'))
abba = importlib.util.module_from_spec(spec); spec.loader.exec_module(abba)


def telemetry(state, sticky, legacy=10):
    return f'[Qwen36DecodeSummary] native_moe_numa_sticky_state={state} native_moe_numa_sticky_dispatch_ops={sticky} native_moe_numa_legacy_dispatch_ops={legacy} native_moe_numa_node_dispatch_ops=0:5,1:5'


class DispatchProofTest(unittest.TestCase):
    def test_off_requires_live_debug_disable_and_no_sticky_dispatch(self):
        self.assertTrue(abba.verify_dispatch_policy(telemetry('disabled_by_debug', 0), 'off')['verified'])
        for text in (telemetry('disabled_by_policy', 0), telemetry('enabled', 3),
                telemetry('disabled_by_debug', 0, 0),
                telemetry('disabled_by_debug', 0) + '\n[NUMA] MoE sticky decode dispatch active (nodes=2)'):
            with self.subTest(text=text), self.assertRaises(ValueError): abba.verify_dispatch_policy(text, 'off')

    def test_on_requires_real_qualified_nondegenerate_dispatch(self):
        self.assertTrue(abba.verify_dispatch_policy(telemetry('enabled', 20), 'on')['verified'])
        for state, count in [('enabled', 0), ('enabled_degenerate', 20), ('disabled_unqualified_model', 0), ('single_node', 0)]:
            with self.subTest(state=state), self.assertRaises(ValueError): abba.verify_dispatch_policy(telemetry(state, count), 'on')

    def test_missing_or_partial_request_evidence_fails(self):
        with self.assertRaises(ValueError): abba.verify_dispatch_policy('sticky=1', 'on')
        with self.assertRaises(ValueError): abba.verify_dispatch_policy(telemetry('enabled', 5), 'on', minimum_records=2)

    def test_both_existing_controls_are_set_in_fresh_process(self):
        self.assertEqual(abba.dense_controls('off'), {'DENSECORE_EXPERIMENTAL_MOE_NUMA_STICKY': '0', 'DENSECORE_DEBUG_DISABLE_MOE_NUMA_STICKY': '1'})
        self.assertEqual(abba.dense_controls('on'), {'DENSECORE_EXPERIMENTAL_MOE_NUMA_STICKY': '1', 'DENSECORE_DEBUG_DISABLE_MOE_NUMA_STICKY': '0'})

    def test_release_build_without_debug_control_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'CMakeCache.txt'
            path.write_text('DENSECORE_ENABLE_DEBUG_ENV:BOOL=OFF\n')
            with self.assertRaises(ValueError): abba.require_debug_build({'dense_build_cache': str(path)})
            path.write_text('DENSECORE_ENABLE_DEBUG_ENV:BOOL=ON\n')
            result = abba.require_debug_build({'dense_build_cache': str(path)})
            self.assertTrue(result['diagnostic_env_enabled'])
            self.assertEqual(len(result['sha256']), 64)

    def test_llama_never_inherits_densecore_ggml_library_or_preload(self):
        config = {'model': '/model.gguf', 'context': 2048, 'llama_command': ['/baseline/bin/llama-server'],
            'env': {'LD_LIBRARY_PATH': '/dense/build', 'LD_PRELOAD': '/dense/build/libdensecore.so'},
            'dense_env': {'LD_LIBRARY_PATH': '/dense/build'}}
        with patch.dict(os.environ, {'LD_PRELOAD': '/unrelated/library.so'}, clear=True):
            baseline = abba.build_environment(config, 'llama', 16)
            dense = abba.build_environment(config, 'off', 16)
        self.assertEqual(baseline['LD_LIBRARY_PATH'], '/baseline/bin')
        self.assertNotIn('LD_PRELOAD', baseline)
        self.assertEqual(dense['LD_LIBRARY_PATH'], '/dense/build')
        self.assertEqual(dense['DENSECORE_DEBUG_DISABLE_MOE_NUMA_STICKY'], '1')
        self.assertEqual(dense['OMP_NUM_THREADS'], '16')


if __name__ == '__main__': unittest.main()
