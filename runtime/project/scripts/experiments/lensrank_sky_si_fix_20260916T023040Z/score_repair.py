"""Frozen-score comparison after the sky-unit repair, without model selection."""
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits

import repair_experiment as r
import lensrank_main_results_audit_20260916 as audit


def sky_scores(group, rows, module, temperature, corrected):
    from ligo.skymap.io.fits import read_sky_map
    bank = []
    for item, row in zip(group["records"], rows):
        if corrected:
            if r.sha(row["moc_path"]) != row["moc_sha256"]:
                raise RuntimeError("New native MOC changed")
            prob = module.raster_probability(read_sky_map(row["moc_path"], moc=True), 512)
        elif "old_dense" in item:
            prob = np.load(item["old_dense"], mmap_mode="r")
        else:
            prob = module.raster_probability(read_sky_map(item["old_moc"], moc=True), 512)
        if group["deployment"] != "o4b":
            # Original confirmation persisted raw float32 before temperature.
            prob = np.asarray(prob, dtype=np.float32)
        bank.append(module.apply_temperature(prob, temperature).astype(np.float32))
    bank = np.stack(bank)
    if group["deployment"] != "o4b":
        return module.gpu_pair_features(bank)
    normalizer = np.asarray([np.asarray(p, dtype=np.float64).sum() for p in bank])
    n, npix = bank.shape
    overlap, bc = np.zeros((n, n), dtype=np.float64), np.zeros((n, n), dtype=np.float64)
    with threadpool_limits(limits=4):
        for first in range(0, npix, 16384):
            block = np.asarray(bank[:, first:first + 16384], dtype=np.float64) / normalizer[:, None]
            overlap += block @ block.T
            sq = np.sqrt(block)
            bc += sq @ sq.T
    ii, jj = np.triu_indices(n, 1)
    return np.log(np.maximum(npix * overlap[ii, jj], 1e-300)), bc[ii, jj].clip(0, 1), {"accumulation": "archived O4b float64 pixel blocks", "n_events": n}


def run(root):
    c = r.verify(root)
    if not (root / "MAPS_COMPLETE.json").exists():
        raise RuntimeError("Do not score partial catalogs")
    frame = pd.read_parquet(root / "tables/map_events.parquet")
    sky = r.load(r.FIX, "repair_pair_sky")
    rows, effects, replays, invariant = [], [], [], []
    out = root / "results"
    for g in c["groups"]:
        events = frame[frame.group == g["id"]].sort_values("idx").to_dict("records")
        if len(events) != len(g["records"]):
            raise RuntimeError("Incomplete event group")
        ii, jj = np.triu_indices(len(events), 1)
        cache = {}
        for spec in g["recipes"]:
            temp = float(spec["temperature"])
            if temp not in cache:
                old, oldbc, oa = sky_scores(g, events, sky, temp, False)
                new, newbc, na = sky_scores(g, events, sky, temp, True)
                cache[temp] = old, oldbc, new, newbc
                r.write(out / g["id"] / f"NUMERICAL_T{temp:g}.json", {"old": oa, "new": na})
            old, oldbc, new, newbc = cache[temp]
            f = pd.read_parquet(spec["pair_path"])
            if not np.array_equal(f.idx_i, ii) or not np.array_equal(f.idx_j, jj):
                raise RuntimeError("Pair order mismatch")
            if g["deployment"] == "o4b":
                wf = f.waveform_score.to_numpy(float)
            else:
                config = spec["joint_config"]
                wf = f.retained_OMC_waveform.to_numpy(float) + config["gamma"] * f.joint_penalty.to_numpy(float) + config["beta"] * f.joint_increment.to_numpy(float)
            old_wf_bytes, old_time_bytes = wf.tobytes(), f.time_score.to_numpy(float).tobytes()
            weights = np.asarray(spec["upstream_weights"], float)
            archived = f.sky_raw_log_bf.to_numpy(float)
            oldscore = np.c_[wf, f.time_score, archived] @ weights
            replay = np.c_[wf, f.time_score, old] @ weights
            newscore = np.c_[wf, f.time_score, new] @ weights
            key = {"deployment": g["deployment"], "catalog": g["catalog"], "slot": spec["slot"], "seed": spec["seed"]}
            ma, mr = audit.metrics(f, oldscore), audit.metrics(f, replay)
            same_metrics = all(abs(ma[k] - mr[k]) <= 1e-12 for k in audit.COLS)
            replay_row = {**key, "max_abs_sky_replay_difference": float(np.max(abs(old - archived))),
                          "all_archived_fusion_metrics_exact": same_metrics}
            replays.append(replay_row)
            if not same_metrics:
                pd.DataFrame(replays).to_csv(root / "tables/REPLAY_FAILURE.csv", index=False)
                raise RuntimeError("Original-score replay altered metrics; cannot attribute all differences to unit repair")
            for mode, value in [("waveform_frozen", wf), ("time_frozen", f.time_score),
                                ("sky_archived", archived), ("sky_SI_fixed", new),
                                ("fusion_archived", oldscore), ("fusion_SI_fixed", newscore)]:
                rows.append({**key, "method": mode, **audit.metrics(f, value)})
            true = f.is_true_pair.to_numpy(bool)
            for name, mask in [("all", np.ones(len(f), bool)), ("companion", true), ("null", ~true),
                               ("null_old_positive_tail1pct", (~true) & (archived >= np.quantile(archived[~true], .99)))]:
                d = new[mask] - archived[mask]
                effects.append({**key, "population": name, "pairs": int(mask.sum()), "median_signed_delta": float(np.median(d)),
                                "median_abs_delta": float(np.median(abs(d))), "p90_abs_delta": float(np.quantile(abs(d), .9)),
                                "p99_abs_delta": float(np.quantile(abs(d), .99)), "max_abs_delta": float(np.max(abs(d))),
                                "sign_flip_fraction": float(np.mean(np.sign(new[mask]) != np.sign(archived[mask]))),
                                "spearman": float(spearmanr(new[mask], archived[mask]).statistic)})
            keep = [x for x in ["idx_i", "idx_j", "event_i", "event_j", "is_true_pair", "true_pair_family", "event_count", "time_score"] if x in f]
            result = f[keep].copy()
            result["waveform_score_frozen"] = wf
            result["sky_raw_log_bf_archived"] = archived
            result["sky_raw_log_bf_SI_fixed"] = new
            result["sky_BC_SI_fixed"] = newbc
            result["final_score_archived"] = oldscore
            result["final_score_SI_fixed"] = newscore
            result["posterior_temperature_frozen"] = temp
            for j, name in enumerate(["waveform", "time", "sky"]):
                result[f"{name}_contribution_SI_fixed"] = np.c_[wf, f.time_score, new][:, j] * weights[j]
            dest = out / g["id"] / f"model_{spec['slot']}"
            dest.mkdir(parents=True, exist_ok=True)
            result.to_parquet(dest / "paired_scores.parquet", index=False)
            unchanged = old_wf_bytes == wf.tobytes() and old_time_bytes == result.time_score.to_numpy(float).tobytes()
            if not unchanged:
                raise RuntimeError("Frozen channel mutated")
            invariant.append({**key, "waveform_time_bitwise_unchanged": unchanged, "weights_unchanged": True, "temperature_unchanged": True})
            print(json.dumps({"score_complete": g["id"], "model": spec["slot"], "R10_old": ma["R10"], "R10_new": rows[-1]["R10"]}), flush=True)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(root / "tables/metrics_per_catalog_model.csv", index=False, encoding="utf-8-sig")
    summary = audit.aggregate(metrics, ["deployment", "method"], ["slot"])
    summary.to_csv(root / "tables/metrics_summary.csv", index=False, encoding="utf-8-sig")
    for name, data in [("sky_pair_changes", effects), ("old_score_replay", replays), ("frozen_invariants", invariant)]:
        pd.DataFrame(data).to_csv(root / f"tables/{name}.csv", index=False, encoding="utf-8-sig")
    r.verify(root)
    r.write(root / "RESULT_STATUS.json", {"state": c["final_status"], "maps_complete": len(frame), "groups": len(c["groups"]),
                                          "code_unit_fix_complete": True, "fixed_weight_comparison_complete": True,
                                          "new_temperature_or_weights_selected": False, "historical_inputs_unchanged": True,
                                          "real_scores_ranks_PE_unchanged_by_construction": True,
                                          "new_blind_test": False, "official_or_PE_based_selection": False})
    report(root, summary, frame)


def report(root, summary, events):
    lines = ["# GWLR-SKY-SI-FIX-01 修复与冻结系数对照", "",
             "状态：HOLD_FOR_AUTHOR_REVIEW_NO_ADOPTION_NO_OVERWRITE。历史归档、论文和真实候选表未覆盖。", "",
             "## 修复范围", "",
             "仅修复注入BAYESTAR低层自旋转换的质量单位：太阳质量乘lal.MSUN_SI后传入接口。",
             "事件源参数、PSD、测量随机种子、运行期时间安排、波形/时间分数、天空温度和融合权重全部保持原值。",
             "本轮是同一已打开数据上的bug影响对照，不是新盲测、没有重训encoder、没有重新选权。",
             "O3/O4a使用原三个190事件目录和三个冻结模型；O4b使用三个原450事件目录。",
             "未取得O4b500份190事件子目录UID清单，因此未声称重现该项复算。", "",
             "## 检索指标", "",
             "|运行期|版本|R@1|R@10|AP|F50|F90|", "|---|---|---|---|---|---|---|"]
    for dep in ("gwtc3", "gwtc4", "o4b"):
        for mode in ("waveform_frozen", "time_frozen", "sky_archived", "sky_SI_fixed", "fusion_archived", "fusion_SI_fixed"):
            row = summary[(summary.deployment == dep) & (summary.method == mode)].iloc[0]
            values = [f"{row[k+'_mean']:.6f} +/- {row[k+'_std']:.6f}" for k in ["R1", "R10", "AP", "F50", "F90"]]
            lines.append("|" + "|".join([dep, mode, *values]) + "|")
    lines += ["", "SD口径：每个模型先平均其目录，再计算三个模型之间样本SD；不是总体置信区间。",
              "", "## 真实候选与后续边界", "",
              "真实公开PE天空图不经过修复函数。本轮真实三通道分数和权重均未改变，故真实排名、PE与官方重合计数保持原值，不应宣传为新提升。",
              "原真实结果：O3 Top10 Mc BC>=0.5为10/10、Dmax<=3为10/10、官方1%前端7、公开Hanabi重合6；O4a分别10/10、10/10、8、6；O4b前两项10/10、10/10，官方信息未有可核验输入。",
              "这些真实PE/官方统计沿用既有审核，未重做PE或Hanabi，也不参与本轮决策。",
              "原天空温度可能吸收了错误模拟的部分偏差。本轮冻结它以隔离单位修复；不能据此声称完成了新版本validation重新校准。",
              "需要重新校准时应另建明确对照，仍只用validation，不能按真实PE或本轮测试优劣选方案。",
              "当前条件Gaussian触发量、HL注入与部分真实HLV网络差异、O3历史日历以及上游适应性开发的限制仍然保留。",
              "", "## 可复现文件", "",
              "contracts/ANALYSIS_CONTRACT.json含冻结输入路径、SHA256、所有事件及配置；pilot含旧实现回放；maps保存原生MOC；results含逐pair新旧分数；tables含逐seed指标、长尾和不变量。",
              "所有存量输入在结束时再次验证SHA256，波形/时间数组逐字节检查不变。",
              "", f"共完成{len(events)}张事件级天空图，不为O3/O4a三个模型重复生成同一事件地图。"]
    (root / "reports/REPAIR_AND_FIXED_SCORE_RESULTS_CN.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    run(parser.parse_args().root)
