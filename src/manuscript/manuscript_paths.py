"""Portable data locations for the manuscript workflow snapshots."""
import os
from pathlib import Path


def workspace_root():
    return Path(os.environ.get('THERMOSTERIC_WORKSPACE', 'outputs')).expanduser().resolve()


def input_path(variable, relative):
    return Path(os.environ.get(variable, str(workspace_root() / relative))).expanduser().resolve()
