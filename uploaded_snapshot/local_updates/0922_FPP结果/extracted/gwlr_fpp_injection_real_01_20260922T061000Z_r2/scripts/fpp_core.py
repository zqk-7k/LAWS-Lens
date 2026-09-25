"""Conditional empirical tails. Shared-event pairs are not iid trials."""
import numpy as np


def tail_counts(background, query):
    background = np.asarray(background, dtype=float)
    query = np.asarray(query, dtype=float)
    if background.ndim != 1 or not len(background) or not np.isfinite(background).all():
        raise ValueError('Invalid background')
    if not np.isfinite(query).all():
        raise ValueError('Invalid query scores')
    return len(background) - np.searchsorted(np.sort(background), query, side='left')


def assert_disjoint(background, evaluation, fields):
    counts = {}
    for name in fields:
        if background[name].isna().any() or evaluation[name].isna().any():
            raise ValueError('Missing grouping key: '+name)
        counts[name] = len(set(background[name].astype(str)) & set(evaluation[name].astype(str)))
        if counts[name]:
            raise ValueError('Calibration/evaluation overlap: '+name)
    return counts


class ConditionalNull:
    def __init__(self, scores, pair_i, pair_j, source_uids, noise_uids):
        self.scores = np.asarray(scores, float)
        self.i, self.j = np.asarray(pair_i, int), np.asarray(pair_j, int)
        self.sources, self.noise = np.asarray(source_uids, str), np.asarray(noise_uids, str)
        n = len(self.sources)
        if n < 2 or len(set(self.sources)) != n or len(self.noise) != n:
            raise ValueError('Require distinct null sources with noise identifiers')
        if len(self.scores) != n*(n-1)//2 or len(self.i) != len(self.scores) or len(self.j) != len(self.scores):
            raise ValueError('Incomplete all-pair background')
        if ((self.i < 0) | (self.j >= n) | (self.i >= self.j)).any():
            raise ValueError('Noncanonical or invalid pair')
        if len(set(zip(self.i, self.j))) != len(self.i) or not np.isfinite(self.scores).all():
            raise ValueError('Duplicate pair or invalid score')
        self.order = np.argsort(-self.scores, kind='stable')
        seen_s, seen_n = set(), set()
        source_count, noise_count = [0], [0]
        for row in self.order:
            seen_s.update((self.sources[self.i[row]], self.sources[self.j[row]]))
            seen_n.update((self.noise[self.i[row]], self.noise[self.j[row]]))
            source_count.append(len(seen_s))
            noise_count.append(len(seen_n))
        self.prefix_sources, self.prefix_noise = np.array(source_count), np.array(noise_count)

    def query(self, scores, group_sensitivity=True):
        scores = np.asarray(scores, float)
        k = tail_counts(self.scores, scores)
        out = dict(background_exceedances=k,
                   conditional_FPP=k / len(self.scores),
                   tail_distinct_sources=self.prefix_sources[k],
                   tail_distinct_noise_parents=self.prefix_noise[k],
                   zero_exceedance_unresolved=k == 0,
                   low_tail_count=k < 20,
                   score_above_background_max=scores > np.max(self.scores))
        if group_sensitivity:
            for unit, groups in [('source', self.sources), ('noise', self.noise)]:
                lo, hi = np.full(scores.shape, np.inf), np.full(scores.shape, -np.inf)
                for group in np.unique(groups):
                    keep = (groups[self.i] != group) & (groups[self.j] != group)
                    if not keep.any():
                        raise ValueError('Insufficient groups for leave-one-group-out sensitivity')
                    p = tail_counts(self.scores[keep], scores) / keep.sum()
                    lo, hi = np.minimum(lo, p), np.maximum(hi, p)
                out[unit+'_leave_one_out_min'] = lo
                out[unit+'_leave_one_out_max'] = hi
        return out

    def catalogs(self, n_events, n_draws, seed):
        if not 2 <= n_events <= len(self.sources) or n_draws < 1:
            raise ValueError('Invalid subcatalog request')
        rng = np.random.default_rng(seed)
        indices = np.array([rng.choice(len(self.sources), n_events, replace=False)
                            for _ in range(n_draws)])
        matrix = np.full((len(self.sources), len(self.sources)), np.nan)
        matrix[self.i, self.j] = matrix[self.j, self.i] = self.scores
        i, j = np.triu_indices(n_events, 1)
        values = matrix[indices[:, i], indices[:, j]]
        return indices, np.max(values, axis=1)


def catalog_queries(maxima, pair_fpp, thresholds, n_events):
    return dict(conditional_catalog_FPP_mc=tail_counts(maxima, thresholds) / len(maxima),
                conditional_expected_false_pairs=n_events*(n_events-1)/2*np.asarray(pair_fpp),
                catalog_zero_exceedance_unresolved=tail_counts(maxima, thresholds) == 0)


def conservative_cut(scores, alpha):
    if not 0 < alpha < 1:
        raise ValueError('Invalid alpha')
    x = np.sort(np.asarray(scores, float))
    if not len(x) or not np.isfinite(x).all():
        raise ValueError('Invalid scores')
    return float(np.nextafter(x[len(x)-1-int(alpha*len(x))], np.inf))


def interval_union(intervals):
    rows = sorted((float(a), float(b)) for a, b in intervals)
    if not rows or any(not np.isfinite([a, b]).all() or b <= a for a, b in rows):
        raise ValueError('Invalid calendar')
    merged = [list(rows[0])]
    for start, end in rows[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged
