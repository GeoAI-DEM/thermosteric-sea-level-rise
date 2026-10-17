#!/usr/bin/env python3
"""Run numerical unit tests offline, without DEM integration or downloads."""
import importlib.util
import os
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src/manuscript'))
from run import configure_imports


def main():
    configure_imports()
    os.environ.setdefault('THERMOSTERIC_WORKSPACE', str(ROOT / 'outputs/tests'))
    os.environ.setdefault('MPLBACKEND', 'Agg')
    suite = unittest.TestSuite()
    files = sorted((ROOT / 'tests').glob('test_*.py'))
    files += sorted((ROOT / 'src/manuscript').rglob('test_*.py'))
    for index, path in enumerate(files):
        spec = importlib.util.spec_from_file_location('workflow_test_{}'.format(index), str(path))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(module))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    sys.exit(main())
