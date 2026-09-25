#!/usr/bin/env python3
"""Append-only correction of the rejected cross-network numeric-ID assumption."""
import json
from datetime import datetime, timezone
from pathlib import Path
import shutil
import sys
sys.path.insert(0, '/root/autodl-tmp/gw-catalog/scripts/experiments')
import mcwf_temporal_response_20260908 as t
import mcwf_source_time_namespaced_20260908 as corrected


def run():
    r = t.PROJECT / 'results/mcwf_source_group_time_exploratory_20260908T155219Z' if hasattr(t, 'PROJECT') else Path('/root/autodl-tmp/gw-catalog/results/mcwf_source_group_time_exploratory_20260908T155219Z')
    target = r / 'contracts/SOURCE_ID_NAMESPACE_INVALIDITY.json'
    if target.exists():
        raise RuntimeError('Append-only invalidation already exists')
    corrected.cross_namespace_audit(r)
    t.dev.json_write(target, {'UTC': datetime.now(timezone.utc).isoformat(), 'status': 'INVALID_CROSS_NETWORK_SOURCE_ID_ASSUMPTION',
                     'old_files_retained': True, 'numeric_results_not_eligible_for_comparison_or_adoption': True,
                     'invalid_assumption': 'GW-LMC:ET andGW-LMC:2.5PLUS same numeric event_id was treated as same physical source.',
                     'verification': 'All145same-ID source rows differ in mass,sky andspin;full physical fingerprint cross-network matching is audited separately.',
                     'consequence': 'Twelve2.5PLUS systems were incorrectly excluded. The22-system time population and all resulting round11 candidate rankings are INVALID.',
                     'still_valid_finding': '63rawdetectableimagepairdelays camefrom34sources;100000resampledvaluesarenot100000independentsources;finite-sourcebandwidth needs correction.',
                     'new_round': 'MCWF-NAMESPACED-SOURCE-TIME-13,independentdirectory;no replacement ofround11numeric outputs.'})
    (r / 'reports/TIME_NAMESPACE_INVALIDITY_CN.md').write_text(
        '# 本轮时间结果无效：跨网络 ID 命名空间错误\n\n'
        '本轮把 ET 与 2.5PLUS 中相同数字的 event_id 当作同一源，错误删除12个外部系统。'
        '进一步读取 SourceParams 后，145个同号条目的质量、天空方向、自旋均不一致。'
        '因此不能据此声称原先验与当前测试集有12个系统泄漏。此前该重合说法撤回。\n\n'
        '本轮22系统先验和全部排名仅保留为失败实现的完整记录，不得作为正确时间修正的结果。'
        '原34系统产生63对、再重采样100000条导致带宽独立样本数不正确的问题仍然成立，另建13轮重新计算。\n\n'
        'O4a source1568在ET内部命名空间中的历史split异常是另一件事，不由本次跨网络命名空间错误推翻；单独保留审计。\n\n'
        'HOLD_FOR_AUTHOR_REVIEW_NO_ADOPTION_NO_OVERWRITE\n', encoding='utf-8')
    shutil.copy2(__file__, r / 'scripts/invalidate_time_namespace.py')


if __name__ == '__main__':
    run()
