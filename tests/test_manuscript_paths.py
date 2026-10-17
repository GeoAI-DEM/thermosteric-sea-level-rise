import importlib.util
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1] / 'src/manuscript'


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, str(SOURCE / filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


paths = load('manuscript_paths_test', 'manuscript_paths.py')
launcher = load('manuscript_launcher_test', 'run.py')


class ManuscriptPathsTests(unittest.TestCase):
    def test_default_workspace(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(paths.workspace_root(), (Path.cwd() / 'outputs').resolve())

    def test_workspace_and_input_overrides(self):
        with tempfile.TemporaryDirectory() as temp:
            with patch.dict(os.environ, {'THERMOSTERIC_WORKSPACE': temp}, clear=True):
                self.assertEqual(paths.input_path('THERMOSTERIC_GEBCO', 'inputs/dem.tif'), Path(temp) / 'inputs/dem.tif')
                with patch.dict(os.environ, {'THERMOSTERIC_GEBCO': str(Path(temp) / 'other.tif')}):
                    self.assertEqual(paths.input_path('THERMOSTERIC_GEBCO', 'unused'), Path(temp) / 'other.tif')

    def test_reject_external_script(self):
        with self.assertRaises(ValueError):
            launcher.script_path('../../README.md')

    def test_known_script(self):
        path = launcher.script_path('contribution_structure_analysis/calculate_long_term_contributions.py')
        self.assertTrue(path.is_file())

    def test_snapshot_checksums(self):
        root = SOURCE.parents[1]
        manifest = json.loads((root / 'docs/workflow_snapshot_manifest.json').read_text())
        for record in manifest['files']:
            actual = hashlib.sha256((root / record['repository_path']).read_bytes()).hexdigest()
            self.assertEqual(actual, record['repository_sha256'])

    def test_rebuild_dynamic_dependencies_are_source_modules(self):
        code = '''
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from run import configure_imports
configure_imports()
import rebuild_lon_aligned_figures as rebuild
modules = rebuild.configure(Path(sys.argv[2]))
for module in modules.values():
    Path(module.__file__).resolve().relative_to(Path(sys.argv[1]).resolve())
assert len(modules) == 9
'''
        with tempfile.TemporaryDirectory() as temp:
            env = dict(os.environ, THERMOSTERIC_WORKSPACE=temp, MPLBACKEND='Agg')
            subprocess.check_call([sys.executable, '-B', '-c', code, str(SOURCE), temp], env=env)


if __name__ == '__main__':
    unittest.main()
