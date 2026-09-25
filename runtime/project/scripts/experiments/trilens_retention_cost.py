"""Validation-frozen pair retention and measured follow-up accounting.

This module does not estimate PE or Hanabi runtimes from invented constants.
Higher scores must mean stronger selection support for every input method.
"""
import math


def validation_threshold(scores, labels, target_recall=0.9):
    if len(scores) != len(labels) or not scores or not 0 < target_recall <= 1:
        raise ValueError("Invalid arrays or recall target")
    if not all(math.isfinite(float(s)) for s in scores):
        raise ValueError("Nonfinite score")
    if not all(y in (0, 1, False, True) for y in labels):
        raise ValueError("Binary companion labels required")
    positives = sorted((float(s) for s,y in zip(scores,labels) if y), reverse=True)
    if not positives:
        raise ValueError("No validation companion pairs")
    return positives[math.ceil(target_recall * len(positives))-1]


def retained_counts(scores, labels, threshold):
    if len(scores) != len(labels) or not all(math.isfinite(float(s)) for s in scores):
        raise ValueError("Invalid pair scores")
    if not math.isfinite(threshold) or not all(y in (0, 1, False, True) for y in labels):
        raise ValueError("Invalid threshold or labels")
    selected = [float(s) >= threshold for s in scores]
    tp = sum(bool(y) and keep for y,keep in zip(labels,selected))
    fp = sum(not bool(y) and keep for y,keep in zip(labels,selected))
    total_positive = sum(bool(y) for y in labels)
    return dict(true_retained=tp,false_retained=fp,total_selected=tp+fp,
                pair_recall=tp/total_positive if total_positive else None,
                precision=tp/(tp+fp) if tp+fp else None)


def total_measured_cost(preparation, frontend, stage2_costs, stage2_pass, stage3_costs):
    """Costs for the *actual selected pairs*, not all catalog pairs.

    Accept consistent units only: wall time, CPU hours and GPU hours must be
    accounted separately. Sum of job wall times is workload, not parallel makespan.
    """
    if set(stage2_pass) != set(stage2_costs):
        raise ValueError("Every selected pair needs stage-2 result and timing")
    expected = {pair for pair,passed in stage2_pass.items() if passed}
    if set(stage3_costs) != expected:
        raise ValueError("Stage-3 cost must cover exactly the pairs that passed stage 2")
    values = [preparation,frontend,*stage2_costs.values(),*stage3_costs.values()]
    if not all(math.isfinite(v) and v >= 0 for v in values):
        raise ValueError("Measured costs required")
    return sum(values)


def self_test():
    threshold=validation_threshold([1,2,3,4,5],[1,0,1,0,1],.5)
    assert threshold==3
    assert retained_counts([3,3,2,1],[1,0,1,0],threshold)==dict(
        true_retained=1,false_retained=1,total_selected=2,pair_recall=.5,precision=.5)
    assert total_measured_cost(2,3,{"a":5,"b":7},{"a":True,"b":False},{"a":11})==28
    for fn in (lambda:validation_threshold([1],[0]),
               lambda:validation_threshold([float("nan")],[1]),
               lambda:total_measured_cost(0,0,{"a":1},{"a":False},{"a":4})):
        try:
            fn()
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid scientific input accepted")
    return {"status":"PASS", "ties":"all equal-threshold scores retained",
            "not_empirical_recall_results":True,"test_threshold_retuning":False}


if __name__=="__main__":
    import json
    print(json.dumps(self_test()))
