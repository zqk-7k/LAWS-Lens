"""Numerical paper replay and dependency audit without editing the snapshot."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def compare_csv(a, b):
    import numpy as np
    import pandas as pd
    left, right = pd.read_csv(a), pd.read_csv(b)
    if list(left.columns) != list(right.columns) or left.shape != right.shape:
        return dict(status='FAIL_SCHEMA', left_shape=left.shape, right_shape=right.shape)
    diffs, fail = {}, []
    for col in left:
        if pd.api.types.is_numeric_dtype(left[col]) and pd.api.types.is_numeric_dtype(right[col]):
            x, y = left[col].to_numpy(float), right[col].to_numpy(float)
            if not np.allclose(x, y, rtol=1e-10, atol=1e-12, equal_nan=True):
                fail.append(col)
            finite = np.isfinite(x) & np.isfinite(y)
            diffs[col] = float(abs(x[finite]-y[finite]).max()) if finite.any() else 0.
        elif not left[col].fillna('<NA>').equals(right[col].fillna('<NA>')):
            fail.append(col)
    return dict(status='FAIL' if fail else 'PASS', rows=len(left), failed_columns=fail, errors=diffs)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--release', type=Path, required=True)
    a = ap.parse_args()
    R = a.release
    U = R/'uploaded_snapshot'
    paper = U/'paper_snapshot'
    source = paper/'source_data/c_current'
    output = R/'verification/paper'
    output.mkdir(parents=True, exist_ok=False)
    tables = []
    for p in sorted((R/'runtime/completion/tables').glob('*.csv')):
        if (source/p.name).exists():
            tables.append(dict(file=p.name, **compare_csv(source/p.name, p)))
    gpd = U/'local_updates/0923_固定上界GPD'
    work = output/'gpd'
    work.mkdir()
    shutil.copy2(gpd/'calculate.py', work/'calculate.py')
    (output/'0922_GWLR_TAIL02诊断').symlink_to(U/'local_updates/0922_GWLR_TAIL02诊断', target_is_directory=True)
    with (output/'gpd.stdout.log').open('w') as log, (output/'gpd.stderr.log').open('w') as err:
        result = subprocess.run([sys.executable, '-B', str(work/'calculate.py')], stdout=log, stderr=err)
    results = []
    if result.returncode == 0:
        for p in sorted((work/'tables').glob('*.csv')):
            results.append(dict(file=p.name, **compare_csv(p, gpd/'tables'/p.name)))
            if (source/'fixed_endpoint_gpd'/p.name).exists():
                results.append(dict(file='paper/'+p.name, **compare_csv(p, source/'fixed_endpoint_gpd'/p.name)))
    # Execute the archived plotting algorithm with only filesystem destinations relocated.
    plot = gpd/'论文更新/plot_revision.py'
    redraw_paper = output/'redraw_paper'
    (redraw_paper/'figures').mkdir(parents=True)
    for name in ('fig_background_diagnostics.pdf', 'supp_tail_endpoint_diagnostics.pdf'):
        shutil.copy2(paper/'figures'/name, redraw_paper/'figures'/name)
    plotbase = output/'plot'
    plotbase.mkdir()
    tail = U/'local_updates/0922_GWLR_TAIL02诊断/extracted/gwlr_tail02_20260922T113500Z_r2'
    inj = U/'local_updates/0922_FPP结果/extracted/gwlr_fpp_injection_real_01_20260922T061000Z_r2'
    mapping = dict(BASE=plotbase, PROJECT=U/'local_updates', WT=redraw_paper,
                   TAB=work/'tables', RAW=tail, INJ=inj, OUT=plotbase/'figures')
    tree = ast.parse(plot.read_text(encoding='utf-8-sig'))
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in mapping:
                node.value = ast.Call(func=ast.Name(id='Path', ctx=ast.Load()),
                    args=[ast.Constant(str(mapping[name]))], keywords=[])
    relocated = output/'plot_relocated.py'
    relocated.write_text(ast.unparse(ast.fix_missing_locations(tree))+'\n')
    with (output/'plot.stdout.log').open('w') as log, (output/'plot.stderr.log').open('w') as err:
        plotted = subprocess.run([sys.executable, '-B', str(relocated)], stdout=log, stderr=err)
    curve = redraw_paper/'source_data/c_current/fixed_endpoint_gpd/figure4_curves.csv'
    curve_check = compare_csv(curve, source/'fixed_endpoint_gpd/figure4_curves.csv') if curve.exists() else {'status':'NOT_CREATED'}
    figures = []
    scripts = list(U.rglob('*.py')) + list((R/'runtime/completion/scripts').glob('*.py'))
    for tex in ('main.tex', 'supplementary.tex'):
        for rel in re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', (paper/tex).read_text()):
            name = Path(rel).stem
            refs = [str(s.relative_to(R)) for s in scripts if name in s.read_text(errors='replace')]
            figures.append(dict(tex=tex, artifact=rel, exists=(paper/rel).is_file(),
                sha256=sha(paper/rel) if (paper/rel).is_file() else None,
                script_text_references=refs,
                verified_redraw=rel in ('figures/fig_background_diagnostics.pdf','figures/supp_tail_endpoint_diagnostics.pdf') and plotted.returncode==0,
                other_redraw_status='NOT_EXECUTED_OR_NO_CURRENT_ENTRYPOINT'))
    report = dict(snapshot_modified=False, common_C_tables=tables,
        fixed_GPD_returncode=result.returncode, fixed_GPD_comparisons=results,
        figure4_redraw_returncode=plotted.returncode, figure4_curve_comparison=curve_check,
        figures=figures, full_paper_figures_rebuilt=False,
        statistical_validation_of_real_FPP=False, annual_FAR_supported=False,
        fixed_endpoint_posthoc=True, fonts_byte_identical_not_claimed=True)
    (output/'REPORT.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(dict(C_tables=len(tables), C_table_failures=sum(r['status']!='PASS' for r in tables),
        GPD_returncode=result.returncode, GPD_comparisons=len(results),
        GPD_failures=sum(r['status']!='PASS' for r in results), plot_returncode=plotted.returncode,
        curve=curve_check['status'], figures=len(figures))))

if __name__ == '__main__':
    main()
