import argparse
import csv
import json
import contextlib
import io
from pathlib import Path
import tempfile
import unittest

from pipeline import FEATURES, prepare, timestamp
from benchmark_local import score


class PreprocessingTests(unittest.TestCase):
    def test_metric_support_and_missing_classes(self):
        result = score(['1', '1', '2'], ['1', '2', '2'], ['1', '2', '3'])
        self.assertAlmostEqual(result['accuracy'], 2 / 3)
        self.assertAlmostEqual(result['macro_f1'], 4 / 9)
        self.assertEqual(result['per_class']['3']['support'], 0)
        self.assertEqual(result['confusion_matrix'], [[1, 1, 0], [0, 1, 0], [0, 0, 0]])

    def test_timezone_equivalence(self):
        self.assertEqual(timestamp('2018-12-15 09:00:00 +0900'),
                         timestamp('2018-12-15T00:00:00Z'))
        with self.assertRaises(ValueError):
            timestamp('2018-12-15T00:00:00')

    def test_windows_and_split_isolation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labels = root / 'labels.csv'
            raw = root / 'raw.csv'
            with labels.open('w', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(['user_id', 'activity_type_id', 'start', 'finish'])
                for day in range(1, 6):
                    writer.writerow([4, 1, f'2018-12-{day:02}T00:00:00Z',
                                     f'2018-12-{day:02}T00:00:04Z'])
                writer.writerow([4, 2, '2018-12-01T01:00:00Z', '2018-12-01T01:00:00Z'])
            with raw.open('w', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(['user_id', 'datetime', 'x', 'y', 'z'])
                # Reversed input verifies that sorting happens before gap checks.
                for day in reversed(range(1, 6)):
                    for tick in reversed(range(40)):
                        writer.writerow([4, f'2018-12-{day:02}T00:00:0{tick // 10}.{tick % 10}Z', 1, 2, 2])
                writer.writerow([4, '2018-12-01T00:00:00Z', 1, 2, 2])
                writer.writerow([4, '2018-12-01T00:00:04Z', 1, 2, 2])
            with contextlib.redirect_stdout(io.StringIO()):
                prepare(argparse.Namespace(output=str(root), labels=str(labels), raw=str(raw),
                                           user=4, seconds=2, max_gap=.25, min_samples=10))
            report = json.loads((root / 'audit.json').read_text())
            self.assertEqual(report['windows'], 10)
            self.assertEqual(report['duplicate_timestamps'], 1)
            self.assertEqual(report['unmatched_rows'], 1)
            self.assertEqual(report['invalid_label_rows'], [7])
            with (root / 'windows.csv').open(newline='') as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(set(FEATURES) - set(rows[0]), set())
            self.assertEqual(float(rows[0]['magnitude_mean']), 3)
            for column in ('session', 'interval_id'):
                split_sets = {split: {r[column] for r in rows if r['split'] == split}
                              for split in ('train', 'validation', 'test')}
                self.assertFalse(split_sets['train'] & split_sets['validation'])
                self.assertFalse(split_sets['train'] & split_sets['test'])
                self.assertFalse(split_sets['validation'] & split_sets['test'])

    def test_overlap_and_gap_exclusion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labels, raw = root / 'labels.csv', root / 'raw.csv'
            with labels.open('w', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(['user_id', 'activity_type_id', 'start', 'finish'])
                for day in range(1, 4):
                    writer.writerow([4, 1, f'2018-12-{day:02}T00:00:00Z',
                                     f'2018-12-{day:02}T00:00:04Z'])
                writer.writerow([4, 2, '2018-12-01T00:00:01Z', '2018-12-01T00:00:02Z'])
            with raw.open('w', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(['user_id', 'datetime', 'x', 'y', 'z'])
                for day in range(1, 4):
                    for tick in range(40):
                        if day == 3 and 5 <= tick < 15:
                            continue
                        writer.writerow([4, f'2018-12-{day:02}T00:00:0{tick // 10}.{tick % 10}Z', 1, 2, 2])
                writer.writerow([4, '2018-12-02T00:00:00Z', 9, 9, 9])
            with contextlib.redirect_stdout(io.StringIO()):
                prepare(argparse.Namespace(output=str(root), labels=str(labels), raw=str(raw),
                                           user=4, seconds=2, max_gap=.25, min_samples=10))
            report = json.loads((root / 'audit.json').read_text())
            self.assertEqual(report['ambiguous_rows'], 10)
            self.assertEqual(report['conflicting_timestamps'], 1)
            self.assertEqual(report['windows'], 4)
            self.assertEqual(report['excluded_windows'], 2)


if __name__ == '__main__':
    unittest.main()
