"""Validate snapshot accounting against retained, real GB200 allocator evidence."""
import argparse
from pathlib import Path
from memtracker_nccl.phase_memory import analyze_capture
from memtracker_nccl.report_io import read_report, write_report


def validate(fixtures):
    rows=[]
    for record in fixtures['records']:
        report=analyze_capture(record['capture'])
        difference=report['checkpoints'][0]['counter_differences']
        rows.append(dict(allocator_mode=record['allocator_mode'], rank=record['rank'],
            phase=record['phase'], snapshot_sha256=record['snapshot_sha256'],
            counter_differences=difference, reconciled=report['counters_reconciled']))
    return dict(schema_version=1, kind='real_snapshot_accounting_validation',
        snapshots=len(rows), all_counters_reconciled=all(r['reconciled'] for r in rows),
        maximum_absolute_difference_bytes=max(abs(v) for r in rows for v in r['counter_differences'].values()),
        rows=rows, limits='Validates boundary snapshot accounting only, not transient capture or model prediction accuracy.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixtures',type=Path,default=Path('experiments/framework_snapshot_fixtures.json.gz'))
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    report=validate(read_report(args.fixtures))
    write_report(args.output,report)
    print(f"{report['snapshots']} snapshots; max difference {report['maximum_absolute_difference_bytes']} bytes")
    if not report['all_counters_reconciled']:
        raise SystemExit(1)


if __name__=='__main__': main()
