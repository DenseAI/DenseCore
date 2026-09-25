"""Negative controls for the architectural guard; no compiler/dependencies needed."""
import tempfile
from pathlib import Path
import unittest

from check_architecture import check_modules, check_sources


class ArchitectureGuardTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.core = Path(self.temporary.name)

    def write(self, name, text):
        path = self.core / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return path

    def test_backend_rejects_runtime_include_and_state_access(self):
        self.write('src/backend/example.cpp', '#include "densecore/runtime/inference.h"\n'
                   'void run() { GetCurrentWorkContext(); }\n')
        errors = check_sources(self.core)
        self.assertTrue(any('upper-layer header' in e for e in errors))
        self.assertTrue(any('GetCurrentWorkContext' in e for e in errors))

    def test_runtime_utility_and_test_only_hooks_are_distinguished(self):
        self.write('src/backend/example.cpp', '#include "runtime/runtime_env.h"\n'
                   '// GetCurrentBatch must not be used here\n'
                   '#ifdef DENSECORE_TEST_BUILD\nvoid test() { GetCurrentBatch(); }\n#endif\n')
        self.assertEqual(check_sources(self.core), [])
        self.write('src/backend/example.cpp', '#ifdef DENSECORE_TEST_BUILD\nvoid test() {}\n'
                   '#else\nvoid production() { GetCurrentBatch(); }\n#endif\n')
        self.assertTrue(check_sources(self.core))

    def test_implementation_aliases_cannot_hide_multiple_owners(self):
        self.write('src/runtime/implementation.inl', 'void impl() {}\n')
        self.write('src/runtime/a.cpp', '#include "implementation.inl"\n')
        self.write('src/runtime/b.cpp', '#include "runtime/implementation.inl"\n')
        errors = check_sources(self.core)
        self.assertTrue(any('multiple owners' in e for e in errors))

    def test_declaration_header_cannot_include_implementation(self):
        self.write('src/runtime/implementation.inl', 'void impl() {}\n')
        self.write('src/runtime/interface.h', '#include "implementation.inl"\n')
        self.assertTrue(any('declaration header' in e for e in check_sources(self.core)))

    def test_module_assignment_and_variant_inventory_must_match(self):
        self.write('src/backend/example.cpp', 'void example() {}\n')
        build = self.core / 'build'
        self.write('build/densecore-module-map-production.txt',
                   'src/backend/example.cpp|backend_kernels|densecore_backend_kernels_production\n')
        self.assertEqual(check_modules(self.core, build), [])
        self.write('build/densecore-module-map-test.txt',
                   'src/backend/example.cpp|runtime|densecore_runtime_test\n')
        self.assertTrue(any('wrong owner' in e for e in check_modules(self.core, build)))
        self.write('build/densecore-module-map-test.txt', '')
        self.assertTrue(any('inventory differs' in e for e in check_modules(self.core, build)))

    def test_public_compatibility_declaration_does_not_allow_runtime_accessor(self):
        self.write('include/densecore/backend/example.h', 'struct BatchSpec;\nvoid legacy(const BatchSpec*);\n')
        self.assertEqual(check_sources(self.core), [])
        self.write('include/densecore/backend/example.h', 'inline void run() { GetCurrentWorkContext(); }\n')
        self.assertTrue(check_sources(self.core))


if __name__ == '__main__':
    unittest.main()
