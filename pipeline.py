"""Prepare field accelerometer windows and train a reproducible feature baseline."""

import argparse
import bisect
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent
FEATURES = [f'{axis}_{stat}' for axis in ('x', 'y', 'z', 'magnitude')
            for stat in ('mean', 'std', 'min', 'max', 'energy')]


def timestamp(value):
    value = value.strip()
    if value.endswith((' +0900', ' +1000')):
        value = value[:-6] + value[-5:]
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        raise ValueError(f'Timestamp has no timezone: {value}')
    return dt.astimezone(timezone.utc).timestamp()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def prepare(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    intervals = []
    invalid_labels = []
    with Path(args.labels).open(newline='') as stream:
        for index, row in enumerate(csv.DictReader(stream)):
            if row['user_id'] != str(args.user):
                continue
            start, end = timestamp(row['start']), timestamp(row['finish'])
            if end <= start:
                invalid_labels.append(index + 2)
                continue
            intervals.append(dict(id=index, start=start, end=end,
                                  label=row['activity_type_id'], samples=[]))
    intervals.sort(key=lambda item: item['start'])
    if not intervals:
        raise ValueError('No labels found for the requested user')
    starts = [item['start'] for item in intervals]
    report = dict(user=args.user, label_intervals=len(intervals), raw_rows=0,
                  invalid_label_rows=invalid_labels,
                  other_user_rows=0, invalid_rows=0, unmatched_rows=0,
                  ambiguous_rows=0, matched_rows=0, duplicate_timestamps=0,
                  conflicting_timestamps=0, excluded_windows=0,
                  window_seconds=args.seconds, max_gap_seconds=args.max_gap,
                  minimum_samples=args.min_samples)
    first, last = math.inf, -math.inf
    digest = hashlib.sha256()
    # Read bytes once for a reproducible source fingerprint without holding the file in memory.
    with Path(args.raw).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    report['raw_sha256'] = digest.hexdigest()
    report['labels_sha256'] = hashlib.sha256(Path(args.labels).read_bytes()).hexdigest()
    with Path(args.raw).open(newline='') as stream:
        for row in csv.DictReader(stream):
            report['raw_rows'] += 1
            if row['user_id'] != str(args.user):
                report['other_user_rows'] += 1
                continue
            try:
                time = timestamp(row['datetime'])
                values = tuple(float(row[axis]) for axis in ('x', 'y', 'z'))
                if not all(math.isfinite(value) for value in values):
                    raise ValueError('Nonfinite reading')
            except (ValueError, KeyError, TypeError):
                report['invalid_rows'] += 1
                continue
            first, last = min(first, time), max(last, time)
            # Labels use half-open boundaries [start, finish). Check all preceding
            # intervals because labels may overlap or be nested.
            candidates = [item for item in intervals[:bisect.bisect_right(starts, time)]
                          if time < item['end']]
            if len(candidates) > 1:
                report['ambiguous_rows'] += 1
            elif not candidates:
                report['unmatched_rows'] += 1
            else:
                candidates[0]['samples'].append((time, *values))
                report['matched_rows'] += 1
    if math.isfinite(first):
        report['raw_time_range_utc'] = [datetime.fromtimestamp(t, timezone.utc).isoformat()
                                        for t in (first, last)]
    windows, interval_reports = [], []
    for item in intervals:
        by_time = {}
        conflicts = set()
        for sample in item.pop('samples'):
            time = sample[0]
            if time in by_time:
                report['duplicate_timestamps'] += 1
                if by_time[time] != sample:
                    conflicts.add(time)
            by_time[time] = sample
        report['conflicting_timestamps'] += len(conflicts)
        samples = sorted(sample for time, sample in by_time.items() if time not in conflicts)
        differences = [b[0] - a[0] for a, b in zip(samples, samples[1:])]
        interval_reports.append(dict(**item, readings=len(samples),
                                     median_step_seconds=statistics.median(differences)
                                     if differences else None))
        buckets = {}
        for sample in samples:
            bucket = int((sample[0] - item['start']) // args.seconds)
            buckets.setdefault(bucket, []).append(sample)
        full_windows = int((item['end'] - item['start']) // args.seconds)
        for bucket in range(full_windows):
            chunk = buckets.get(bucket, [])
            start = item['start'] + bucket * args.seconds
            gaps = ([chunk[0][0] - start, start + args.seconds - chunk[-1][0]] +
                    [b[0] - a[0] for a, b in zip(chunk, chunk[1:])]) if chunk else []
            if len(chunk) < args.min_samples or max(gaps, default=math.inf) > args.max_gap:
                report['excluded_windows'] += 1
                continue
            # UTC day is a conservative session proxy until recording IDs are available.
            group = datetime.fromtimestamp(start, timezone.utc).date().isoformat()
            record = dict(user_id=args.user, interval_id=item['id'], session=group,
                          start=start, activity_type_id=item['label'], samples=len(chunk))
            channels = [[sample[i] for sample in chunk] for i in (1, 2, 3)]
            channels.append([math.sqrt(sum(v * v for v in sample[1:])) for sample in chunk])
            for axis, values in zip(('x', 'y', 'z', 'magnitude'), channels):
                stats = (statistics.mean(values), statistics.pstdev(values), min(values),
                         max(values), statistics.mean(v * v for v in values))
                record.update({f'{axis}_{name}': value for name, value in
                               zip(('mean', 'std', 'min', 'max', 'energy'), stats)})
            windows.append(record)
    # Union days connected by an interval so midnight cannot split related windows.
    days = sorted({row['session'] for row in windows})
    parent = {day: day for day in days}
    def find(day):
        while parent[day] != day:
            day = parent[day]
        return day
    interval_days = {}
    for row in windows:
        previous = interval_days.setdefault(row['interval_id'], row['session'])
        parent[find(row['session'])] = find(previous)
    groups = sorted({find(day) for day in days})
    if len(groups) < 3:
        write_json(output / 'audit.json', dict(report, intervals=interval_reports))
        raise ValueError('At least three independent recording-day groups are required; see audit.json')
    test_count = max(1, round(len(groups) * .2))
    val_count = max(1, round(len(groups) * .2))
    train_end = max(1, len(groups) - test_count - val_count)
    mapping = {group: ('train' if i < train_end else 'validation'
                       if i < len(groups) - test_count else 'test')
               for i, group in enumerate(groups)}
    counts = {}
    for row in windows:
        row['split'] = mapping[find(row['session'])]
        classes = counts.setdefault(row['split'], {})
        label = row['activity_type_id']
        classes[label] = classes.get(label, 0) + 1
    with (output / 'windows.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(windows[0]))
        writer.writeheader()
        writer.writerows(windows)
    report.update(windows=len(windows), split_class_counts=counts,
                  session_proxy='UTC recording day; days linked by an interval stay together',
                  split_groups={group: mapping[group] for group in groups}, intervals=interval_reports)
    write_json(output / 'audit.json', report)
    print(json.dumps({key: value for key, value in report.items() if key != 'intervals'}, indent=2))


def train(args):
    import joblib
    import pandas as pd
    import sklearn
    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                                 classification_report, confusion_matrix, f1_score)
    frame = pd.read_csv(Path(args.output) / 'windows.csv')
    parts = {split: frame[frame['split'] == split] for split in ('train', 'validation', 'test')}
    if any(part.empty for part in parts.values()):
        raise ValueError('Every split must contain windows')
    for left, right in (('train', 'validation'), ('train', 'test'), ('validation', 'test')):
        for column in ('session', 'interval_id'):
            if set(parts[left][column]) & set(parts[right][column]):
                raise ValueError(f'Overlapping {column} between {left} and {right}')
    target = 'activity_type_id'
    labels = sorted(frame[target].unique().tolist())
    def evaluate(model, part):
        actual = part[target]
        predicted = model.predict(part[FEATURES])
        return dict(accuracy=accuracy_score(actual, predicted),
                    balanced_accuracy=balanced_accuracy_score(actual, predicted),
                    macro_f1=f1_score(actual, predicted, labels=labels, average='macro', zero_division=0),
                    weighted_f1=f1_score(actual, predicted, average='weighted', zero_division=0),
                    per_class=classification_report(actual, predicted, labels=labels, output_dict=True, zero_division=0),
                    confusion_matrix=confusion_matrix(actual, predicted, labels=labels).tolist())
    models = {'majority': DummyClassifier(strategy='most_frequent'),
              'forest': RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                               random_state=42, n_jobs=-1),
              'forest_weighted': RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                                        class_weight='balanced', random_state=42, n_jobs=-1)}
    results = {}
    for name, model in models.items():
        model.fit(parts['train'][FEATURES], parts['train'][target])
        results[name] = evaluate(model, parts['validation'])
    selected = max(results, key=lambda name: results[name]['macro_f1'])
    destination = Path(args.output)
    audit = json.loads((destination / 'audit.json').read_text(encoding='utf-8'))
    joblib.dump(dict(model=models[selected], features=FEATURES, labels=labels,
                    preprocessing=audit, seed=42,
                    versions=dict(pandas=pd.__version__, sklearn=sklearn.__version__,
                                  joblib=joblib.__version__)), destination / 'baseline.joblib')
    write_json(destination / 'metrics.json', dict(seed=42, labels=labels, validation=results,
               selected=selected, test=evaluate(models[selected], parts['test']),
               note='Macro F1 uses the full observed class set, including classes missing from a split.'))
    print(f'Selected {selected}; results saved to {destination / "metrics.json"}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'train'))
    parser.add_argument('--raw', default=str(ROOT / 'raw_field_acc_user04.csv'))
    parser.add_argument('--labels', default=str(ROOT / 'field_label_train.csv'))
    parser.add_argument('--user', type=int, default=4)
    parser.add_argument('--seconds', type=float, default=2)
    parser.add_argument('--max-gap', type=float, default=.25)
    parser.add_argument('--min-samples', type=int, default=10)
    parser.add_argument('--output', default=str(ROOT / 'outputs'))
    args = parser.parse_args()
    if args.seconds <= 0 or args.max_gap <= 0 or args.min_samples < 2:
        parser.error('Window length and gap must be positive; minimum samples must be at least two')
    (prepare if args.command == 'prepare' else train)(args)


if __name__ == '__main__':
    main()
