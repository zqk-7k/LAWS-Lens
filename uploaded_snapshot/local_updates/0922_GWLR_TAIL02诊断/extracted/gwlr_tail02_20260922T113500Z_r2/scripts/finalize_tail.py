"""Add explicit extrapolation findings without overwriting the first diagnostic package."""
import hashlib
import json
from pathlib import Path
import sys
import tarfile
import pandas as pd

root=Path(sys.argv[1])
support=pd.read_csv(root/'results/REAL_TOP10_EXTRAPOLATION_SUPPORT.csv')
fits=pd.read_csv(root/'results/gpd_fits.csv')
rows=[]
for run in ('O3','O4a','O4b'):
    f=fits[(fits.run==run)&(fits.model=='mean_S')]
    held=pd.read_csv(root/f'inputs/{run}_check_scores.csv').mean_S
    for r in f.itertuples():
        k=int((held>r.endpoint).sum()) if pd.notna(r.endpoint) else 0
        rows.append(dict(run=run,q=r.q,shape=r.shape,endpoint=r.endpoint,
            heldout_above_model_endpoint=k,heldout_maximum=held.max(),
            interpretation='Endpoint sensitivity diagnostic, not an extra post-hoc selection gate'))
pd.DataFrame(rows).to_csv(root/'results/HELDOUT_ENDPOINT_DIAGNOSTIC.csv',index=False,encoding='utf-8-sig')
msg='''# 最先阅读：当前结果不是已完成的匹配背景补充

版本 GWLR-TAIL-02，2026-09-22。未改模型、权重、候选排名或论文。

## 当前实际完成

- 完成 O3/O4a/O4b 的12种冻结评分统计量、36次GPD拟合和独立源/噪声分割的旧test诊断。
- 完成删除最大pair、逐源、逐噪声块敏感性，以及4800次节点/噪声重采样敏感性计算。
- 完成噪声储备盘点、官方O4a背景重建计划、逐项缺失输入表。
- 原经验FPP与计数不变。新增已评分背景事件数 **0**，没有完成大规模匹配背景生成。

## 为什么不能填入正式候选GPD-FPP

背景q98、q99、q99.5范围内的预设数值/预测筛查通过，但全部30个真实Top-10
都超过本轮最高预测检查分数。更重要的是：

- O4b Rank 1 的 S=2.984595，高于三个起点GPD拟合出的有限上端点。
- O4a 前六个候选同样高于三个拟合上端点。
- 直接将模型代入会产生零尾概率。这不是证明其物理背景概率为零，
  而是有限数据下外推无法作为可信候选显著性。
- 原test中也检查了超出拟合上端点的背景计数，见 HELDOUT_ENDPOINT_DIAGNOSTIC.csv。

不依据真实候选结果改换分布、挑阈值或截断负形状参数。不能为了写出不同的
小数强制改用无界指数尾。所有GPD数值仅作背景可行性诊断，不发布候选估计。

## 补背景的实际缺项

排除C所用父块和已用时间段后，原库中尚余候选噪声父块：O3为2，O4a为15，
O4b为43。这些还不是DQ和独立性已认证的新背景。

官方O4a元数据含254个事件，按共享/重叠噪声组成179个连通分量；
哈希划分得到138个fit和116个validation事件。此划分只是新重建计划，
不是说官方原实验采用此划分。254个事件的原清洗后PSD均不在指定位置，
O4a的已检测样本也不能直接充当O3/O4b的总体。

现有真实天空来自公开PE，模拟天空来自HL BAYESTAR。新增真实条件FPP之前，
需补运行期条件化源/选择模型、独立真实噪声及天空处理可转移性对照。
当前已向作者询问是否允许少量相干天空PE对照；若只允许BAYESTAR背景，
交付必须限定为模拟条件FPP，不能声称已经校准真实GWTC显著性。

年度FAR仍不具备有效背景曝光；没有新增次/年数值。

状态：HOLD_MATCHED_BACKGROUND_AND_REAL_CALIBRATION_INCOMPLETE。
当前没有后台批量模拟任务。诊断已完成，后续任务未冒充完成。
'''
with (root/'reports/READ_FIRST_TAIL02_CN.md').open('x',encoding='utf-8') as f:f.write(msg)
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()
original=pd.read_csv(root/'manifests/INPUT_SHA256.csv')
assert all(sha(Path(r.path))==r.sha256 for r in original.itertuples())
files=sorted(p for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
manifest=[dict(path=str(p.relative_to(root)),sha256=sha(p),bytes=p.stat().st_size) for p in files]
pd.DataFrame(manifest).to_csv(root/'manifests/FINAL_SHA256.csv',index=False)
files.append(root/'manifests/FINAL_SHA256.csv')
package=root.with_name(root.name+'_diagnostic_deliverables_v2.tar.gz')
assert not package.exists()
with tarfile.open(package,'w:gz') as tar:
    for p in files:tar.add(p,arcname=str(Path(root.name)/p.relative_to(root)),recursive=False)
expected={str(Path(root.name)/r['path']):r['sha256'] for r in manifest}
with tarfile.open(package) as tar:
    for m in tar.getmembers():
        if m.name in expected:
            assert hashlib.sha256(tar.extractfile(m).read()).hexdigest()==expected.pop(m.name)
assert not expected
digest=sha(package)
package.with_suffix('.gz.sha256').write_text(digest+'  '+package.name+'\n')
delivery=dict(package=str(package),sha256=digest,payload_files=len(manifest),bytes=package.stat().st_size,
    report=str(root/'reports/READ_FIRST_TAIL02_CN.md'),new_scored_background_events=0,
    candidate_GPD_FPP_published=False,state='HOLD_MATCHED_BACKGROUND_AND_REAL_CALIBRATION_INCOMPLETE')
with package.with_suffix('.DELIVERY.json').open('x') as f:json.dump(delivery,f,indent=2)
print(json.dumps(delivery),flush=True)
