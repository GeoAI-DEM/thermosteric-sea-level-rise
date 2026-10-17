#!/usr/bin/env python3
"""Curate study scripts into the portable manuscript snapshot, excluding data."""
import argparse
import ast
import copy
import hashlib
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FOLDERS = (
    'contribution_structure_analysis',
    'contribution_structure_analysis/en4_uniform_recalculation',
    'contribution_structure_analysis/nearest_basin_reassignment_20261015',
    'contribution_structure_analysis/step3_nearest_20261015',
    'figures/step2_interannual_2025_20261015',
    'figures/step2_interannual_2025_nearest_20261015',
    'figures/step3_robustness_nearest_20261015',
    'figures/step4_cross_product_preview_20261015',
    'figures/introduction_logic_redesign_20261003',
    'figures/introduction_schematic_20261003',
)
FIGURES = (
    'aggregate_15arcsec_nested_latitude.py', 'plot_latitude_shell_ridgelines_2025.py',
    'plot_nested_latitude_15arcsec_2025.py', 'plot_thermal_expansion_with_uncertainty.py',
    'plot_ocean_basin_historical_partition.py', 'rebuild_lon_aligned_figures.py',
    'plot_latitudinal_thermosteric_partition_redesign.py',
)
PROJECTIONS = (
    'run_global_projection.py', 'run_sparse_bsts_projection.py',
    'run_performance_weighted_ensemble.py', 'run_depth_band_ensemble_projection.py',
    'run_explainable_dynamic_projection.py',
)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--refresh', action='store_true', help='Explicitly refresh generated snapshot code')
    args = parser.parse_args()
    original = args.source_root.resolve()
    family = original.parent
    target = REPO / 'src/manuscript'
    files = [path for folder in FOLDERS for path in (original / folder).glob('*.py')]
    files += [original / 'figures' / name for name in FIGURES]
    files += [original / 'short_term_projection' / name for name in PROJECTIONS]
    files += [original / 'uncertainty_analysis/tail_uncertainty_proxy/build_tail_uncertainty_proxy.py', original / 'new.py']
    mapping = {
        str(original): 'workspace_root()',
        str(family / 'Average/annual_averages'): "input_path('THERMOSTERIC_ANNUAL_DIR', 'inputs/annual_averages')",
        str(family / 'runV2/GEBCO_2026_Ocean.tif'): "input_path('THERMOSTERIC_GEBCO', 'inputs/GEBCO_ocean.tif')",
        str(family / 'World_Seas_IHO_v3'): "input_path('THERMOSTERIC_BASINS', 'inputs/IHO_ocean_basins')",
    }

    def expression(value):
        for prefix, result in mapping.items():
            if value == prefix:
                return result
            if value.startswith(prefix + '/'):
                return "({} / {!r})".format(result, value[len(prefix) + 1:])
        raise ValueError('Unmapped local input path')

    records = []
    for src in sorted(set(files)):
        raw = src.read_bytes()
        text = raw.decode('utf-8')
        rel = src.relative_to(original)
        dst = target / rel
        if dst.exists() and not args.refresh:
            raise ValueError('Use --refresh to update generated snapshot: ' + str(rel))
        lines = text.splitlines(keepends=True)
        tree = ast.parse(text)
        edits, taken = [], []
        for node in ast.walk(tree):
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                fun = node.value.func
                if (isinstance(fun, ast.Attribute) and fun.attr == 'insert'
                        and isinstance(fun.value, ast.Attribute) and fun.value.attr == 'path'):
                    index = node.lineno - 1
                    indent = lines[index][:node.col_offset]
                    edits.append((sum(map(len, lines[:index])), sum(map(len, lines[:index + 1])),
                                  indent + 'pass  # Source imports are configured by the launcher.\n'))
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'Path'
                    and len(node.args) == 1 and isinstance(node.args[0], ast.Str)
                    and node.args[0].s.startswith(str(family) + '/')):
                index = node.lineno - 1
                begin = sum(map(len, lines[:index])) + node.col_offset
                match = re.match(r"Path\((['\"])(.*?)\1\)", lines[index][node.col_offset:])
                if match is None:
                    raise ValueError('Unsupported Path call: ' + str(rel))
                end = begin + len(match.group(0))
                taken.append((begin, end))
                edits.append((begin, end, expression(node.args[0].s)))
        for node in ast.walk(tree):
            if isinstance(node, ast.Str) and node.s.startswith(str(family) + '/'):
                index = node.lineno - 1
                begin = sum(map(len, lines[:index])) + node.col_offset
                if any(a <= begin < b for a, b in taken):
                    continue
                segment = lines[index][node.col_offset:]
                quote = segment[0]
                if segment[:len(node.s) + 2] != quote + node.s + quote:
                    raise ValueError('Unsupported literal: ' + str(rel))
                edits.append((begin, begin + len(node.s) + 2, 'str({})'.format(expression(node.s))))
        for begin, end, value in sorted(edits, reverse=True):
            text = text[:begin] + value + text[end:]
        parent = "(workspace_root() / {!r})".format(str(rel.parent))
        text = re.sub(r'Path\(__file__\)\.resolve\(\)\.parent(?!s)', lambda match: parent, text)

        def parent_chain(match):
            parts = rel.parts[:-1]
            remaining = parts[:max(0, len(parts) - int(match.group(1)))]
            return 'workspace_root()' if not remaining else "(workspace_root() / {!r})".format('/'.join(remaining))

        text = re.sub(r'Path\(__file__\)\.resolve\(\)\.parents\[(\d+)\]', parent_chain, text)
        modified = ast.parse(text)
        insert = 0
        if modified.body and isinstance(modified.body[0], ast.Expr) and isinstance(modified.body[0].value, ast.Str):
            insert = modified.body[1].lineno - 1 if len(modified.body) > 1 else len(text.splitlines())
        for node in modified.body:
            if isinstance(node, ast.ImportFrom) and node.module == '__future__':
                insert = node.lineno
        parts = text.splitlines(keepends=True)
        parts.insert(insert, 'from manuscript_paths import workspace_root, input_path\n')
        text = ''.join(parts)
        if str(family) in text:
            raise ValueError('Unmapped private path: ' + str(rel))
        final = ast.parse(text)
        before = {node.name: node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
        after = {node.name: node for node in final.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
        # Ignore only the removed import-search mutation, not numerical code.
        class ImportSetupNormalizer(ast.NodeTransformer):
            def visit_Expr(self, node):
                value = node.value
                if (isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute)
                        and value.func.attr == 'insert' and isinstance(value.func.value, ast.Attribute)
                        and value.func.value.attr == 'path'):
                    return ast.Pass()
                return self.generic_visit(node)
        normalized_before = {name: ast.dump(ImportSetupNormalizer().visit(copy.deepcopy(node)))
                             for name, node in before.items()}
        changed = [name for name in before if normalized_before[name] != ast.dump(after[name])]
        allowed = {'main', 'create_optimized_config', 'parse_command_line_args'}
        if str(rel) == 'new.py':
            allowed.add('OptimizedMultiGPUCalculator')
            old = {node.name: node for node in before['OptimizedMultiGPUCalculator'].body if isinstance(node, ast.FunctionDef)}
            new = {node.name: node for node in after['OptimizedMultiGPUCalculator'].body if isinstance(node, ast.FunctionDef)}
            if any(ast.dump(node) != ast.dump(new[name]) for name, node in old.items() if name != '__init__'):
                raise ValueError('A numerical thermal-expansion method changed')
        if set(changed) - allowed:
            raise ValueError('Unexpected non-path changes: {} {}'.format(rel, changed))
        compile(text, str(dst), 'exec')
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(text, encoding='utf-8')
        if src.read_bytes() != raw:
            raise ValueError('Original source changed during synchronization')
        records.append({'source_snapshot_relative_path': str(rel), 'repository_path': str(dst.relative_to(REPO)),
                        'source_sha256': sha(raw), 'repository_sha256': sha(dst.read_bytes()),
                        'adaptation': 'workspace/input defaults and launcher import search only',
                        'functions_with_path_default_changes': changed,
                        'other_function_and_class_asts_unchanged': True})
    (REPO / 'docs/workflow_snapshot_manifest.json').write_text(json.dumps({
        'snapshot_date': '2026-10-17', 'scope': 'code only; original workflow identifiers retained',
        'files': records}, indent=2) + '\n')
    print('Synchronized {} files; original sources unchanged.'.format(len(records)))


if __name__ == '__main__':
    main()
