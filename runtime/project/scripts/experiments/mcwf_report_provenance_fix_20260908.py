#!/usr/bin/env python3
"""Mark an incorrectly reused draft report template without altering metrics."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(root):
    path=root/'reports/TEMPORAL_RESPONSE_ROUND_CN.md'
    target=root/'contracts/REPORT_TEMPLATE_CORRECTION.json'
    if target.exists():raise RuntimeError('Already documented; do not rewrite a sealed report')
    if not path.exists():return
    content=path.read_text(encoding='utf-8')
    before=sha(path)
    notice=('# 已失效的通用报告草稿，不能作为本轮科学说明\n\n'
        '> 下文是 bootstrap 工具误调用第一轮 TEMPORAL 报告模板产生的草稿。'
        '其中方法名称、冻结项和模板化结论不适用于当前轮次，不能引用。'
        '原始数值表未修改；正式方法和结论请阅读同目录 `ROUND_REPORT_CN.md` 与本轮 contracts。'
        '本文件保留是为了记录报告生成问题，不是另一个实验结果。\n\n---\n\n')
    path.write_text(notice+content,encoding='utf-8')
    target.write_text(json.dumps({'UTC':datetime.now(timezone.utc).isoformat(),
        'reason':'Bootstrap utility also generated a first-round-specific generic report.',
        'original_draft_sha256':before,'annotated_draft_sha256':sha(path),
        'authoritative_report':'reports/ROUND_REPORT_CN.md','metrics_modified':False,
        'historical_results_modified':False,'original_draft_content_retained_below_notice':True},indent=2)+'\n')


if __name__=='__main__':
    for value in sys.argv[1:]:run(Path(value))
