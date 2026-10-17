#!/usr/bin/env python3
"""Launch path-configured manuscript calculations without editing their formulas."""
import argparse
import os
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def source_directories():
    return [HERE] + sorted({path.parent for path in HERE.rglob('*.py')
                           if 'tests' not in path.relative_to(HERE).parts})


def configure_imports():
    # Sibling modules are source code; data paths never become import locations.
    for directory in reversed(source_directories()):
        sys.path.insert(0, str(directory))


def script_path(name):
    path = (HERE / name).resolve()
    try:
        path.relative_to(HERE)
    except ValueError:
        raise ValueError('Script must be inside src/manuscript')
    if path.suffix != '.py' or not path.is_file() or path.name in ('run.py', 'manuscript_paths.py'):
        raise ValueError('Unknown workflow script: ' + name)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, default=Path(os.environ.get('THERMOSTERIC_WORKSPACE', 'outputs')),
                        help='Input/output study workspace; never defaults to the authors machine')
    parser.add_argument('script', help='Relative Python script, or list')
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.script == 'list':
        for path in sorted(HERE.rglob('*.py')):
            if path.name not in ('run.py', 'manuscript_paths.py') and not path.name.startswith('test_'):
                print(path.relative_to(HERE))
        return
    path = script_path(args.script)
    workspace = args.workspace.expanduser().resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    os.environ['THERMOSTERIC_WORKSPACE'] = str(workspace)
    os.environ.setdefault('MPLBACKEND', 'Agg')
    configure_imports()
    os.chdir(str(workspace))
    arguments = args.arguments[1:] if args.arguments[:1] == ['--'] else args.arguments
    sys.argv = [str(path)] + arguments
    runpy.run_path(str(path), run_name='__main__')


if __name__ == '__main__':
    main()
