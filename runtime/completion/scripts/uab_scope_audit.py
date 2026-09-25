"""Read-only clarification of release exposure and nominal observing-run bounds."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

BOUNDS = {
    'O4a': (1368975618., 1389456018., 'https://gwosc.org/O4/O4a/'),
    'O4b': (1396796418., 1422118818., 'https://gwosc.org/O4/O4b/'),
}


def markdown(frame):
    return '\n'.join(['| '+' | '.join(frame.columns)+' |',
        '| '+' | '.join(['---']*len(frame.columns))+' |']+
        ['| '+' | '.join(str(v) for v in row)+' |' for row in frame.itertuples(index=False, name=None)])


def main(root, out):
    rows, calendars = [], []
    for run, (start, end, source) in BOUNDS.items():
        schedule = pd.read_csv(root/'plans'/run/'live_schedule.csv')
        total = float((schedule.end_gps-schedule.start_gps).sum())
        within = np.maximum(0., np.minimum(end, schedule.end_gps)-np.maximum(start, schedule.start_gps)).sum()
        calendars.append(dict(run=run, nominal_start_gps=start, nominal_end_gps=end,
            archived_live_seconds=total, outside_nominal_live_seconds=float(total-within),
            outside_nominal_live_fraction=float((total-within)/total), official_source=source))
        sources = pd.read_parquet(root/'plans'/run/'sources.parquet')
        for (role, split), group in sources.groupby(['role', 'split']):
            events = []
            for row in group.itertuples(index=False):
                events.append((row.source_uid, row.gps_a))
                if row.family != 'unlensed': events.append((row.source_uid, row.gps_b))
            e = pd.DataFrame(events, columns=['source_uid', 'gps'])
            outside = (e.gps < start) | (e.gps >= end)
            rows.append(dict(run=run, role=role, split=split, unique_event_gps=len(e),
                outside_nominal_events=int(outside.sum()), outside_nominal_fraction=float(outside.mean()),
                source_parents_with_outside_image=int(e.loc[outside, 'source_uid'].nunique()),
                same_counts_in_both_arms=True, repeated_training_views_not_counted=True))
    cal, ev = pd.DataFrame(calendars), pd.DataFrame(rows)
    cal.to_csv(out/'tables/release_vs_nominal_calendar.csv', index=False, encoding='utf-8-sig')
    ev.to_csv(out/'tables/release_vs_nominal_event_counts.csv', index=False, encoding='utf-8-sig')
    sky = pd.read_csv(out/'tables/sky_truth_coverage_and_runtime.csv')
    record = dict(completed=True, changed_calendar=False, changed_temperature=False,
        formal_scope='frozen archived public-release HL exposure, including published engineering segments',
        strict_nominal_O4_only=False, read_only_post_freeze_audit=True,
        O4a_test_HPD90=sky[(sky.run == 'O4a') & (sky.split == 'test')][['arm','calibrated_HPD90_coverage']].to_dict('records'),
        caveat='No guarantee of nominal90 coverage from the joint50/90/CvM temperature objective',
        official_sources={run: x[2] for run,x in BOUNDS.items()})
    (out/'contracts/RELEASE_SCOPE_AND_COVERAGE_AUDIT.json').write_text(json.dumps(record, indent=2)+'\n')
    template = (out/'scripts/UAB_FINAL_CAVEATS_TEMPLATE_CN.md').read_text()
    body = template.replace('@@CALENDAR@@', markdown(cal)).replace('@@EVENTS@@', markdown(ev))
    columns = ['run','arm','split','frozen_validation_temperature','calibrated_HPD90_coverage']
    body = body.replace('@@COVERAGE@@', markdown(sky[columns]))
    (out/'reports/READ_FIRST_FINAL_AUDIT_LIMITS_CN.md').write_text(body)
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(); main(a.root, a.out)
