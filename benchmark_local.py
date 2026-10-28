"""Measure a dependency-free nearest-centroid baseline on prepared windows."""

import argparse
from collections import Counter
import csv
import ctypes
import json
import math
from pathlib import Path
import statistics
import time

from pipeline import FEATURES, write_json


def score(actual, predicted, labels):
    matrix = [[sum(a == left and p == right for a, p in zip(actual, predicted))
               for right in labels] for left in labels]
    per_class = {}
    for index, label in enumerate(labels):
        support = sum(matrix[index])
        guesses = sum(row[index] for row in matrix)
        tp = matrix[index][index]
        precision = tp / guesses if guesses else 0
        recall = tp / support if support else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
        per_class[label] = dict(precision=precision, recall=recall, f1=f1, support=support)
    return dict(accuracy=sum(a == p for a, p in zip(actual, predicted)) / len(actual),
                macro_f1=statistics.mean(row['f1'] for row in per_class.values()),
                weighted_f1=sum(row['f1'] * row['support'] for row in per_class.values()) / len(actual),
                per_class=per_class, labels=labels, confusion_matrix=matrix)


def peak_memory_mb():
    if not hasattr(ctypes, 'windll'):
        return None
    class Counters(ctypes.Structure):
        _fields_ = [('cb', ctypes.c_ulong), ('faults', ctypes.c_ulong)] + [
            (name, ctypes.c_size_t) for name in ('peak_working', 'working', 'peak_paged',
            'paged', 'peak_nonpaged', 'nonpaged', 'pagefile', 'peak_pagefile')]
    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    get_process = ctypes.windll.kernel32.GetCurrentProcess
    get_process.restype = ctypes.c_void_p
    query = ctypes.windll.psapi.GetProcessMemoryInfo
    query.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]
    if query(get_process(), ctypes.byref(counters), counters.cb):
        return counters.peak_working / 2**20
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent / 'outputs')
    args = parser.parse_args()
    started = time.perf_counter()
    with (args.output / 'windows.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    parts = {split: [row for row in rows if row['split'] == split]
             for split in ('train', 'validation', 'test')}
    if any(not part for part in parts.values()):
        raise ValueError('All three splits must contain windows')
    for column in ('session', 'interval_id'):
        sets = [set(row[column] for row in parts[split]) for split in parts]
        if any(sets[i] & sets[j] for i, j in ((0, 1), (0, 2), (1, 2))):
            raise ValueError(f'Split overlap in {column}')
    def vector(row):
        values = [float(row[name]) for name in FEATURES]
        if not all(math.isfinite(value) for value in values):
            raise ValueError('Features must be finite')
        return values
    inputs = {split: [vector(row) for row in part] for split, part in parts.items()}
    targets = {split: [row['activity_type_id'] for row in part] for split, part in parts.items()}
    labels = sorted(set(label for values in targets.values() for label in values), key=int)
    training_started = time.perf_counter()
    means = [statistics.mean(column) for column in zip(*inputs['train'])]
    scales = [statistics.pstdev(column) or 1 for column in zip(*inputs['train'])]
    normalized = [[(x - mean) / scale for x, mean, scale in zip(row, means, scales)]
                  for row in inputs['train']]
    centroids = {}
    for label in set(targets['train']):
        members = [row for row, actual in zip(normalized, targets['train']) if actual == label]
        centroids[label] = [statistics.mean(column) for column in zip(*members)]
    majority = Counter(targets['train']).most_common(1)[0][0]
    training_seconds = time.perf_counter() - training_started
    def predict(split, model):
        if model == 'majority':
            return [majority] * len(inputs[split])
        result = []
        for row in inputs[split]:
            row = [(x - mean) / scale for x, mean, scale in zip(row, means, scales)]
            result.append(min(sorted(centroids, key=int), key=lambda label:
                          sum((x - center) ** 2 for x, center in zip(row, centroids[label]))))
        return result
    validation = {name: score(targets['validation'], predict('validation', name), labels)
                  for name in ('majority', 'nearest_centroid')}
    selected = max(validation, key=lambda name: validation[name]['macro_f1'])
    test = score(targets['test'], predict('test', selected), labels)
    report = dict(model=selected, validation=validation, test=test,
                  windows={name: len(part) for name, part in parts.items()},
                  fit_seconds=training_seconds, total_seconds=time.perf_counter() - started,
                  peak_process_memory_mb=peak_memory_mb(),
                  note='CPU-only smoke benchmark; random-forest dependencies are not used. '
                       'Macro F1 includes every observed class. Test contains only activities 4 and 5.')
    write_json(args.output / 'local_benchmark.json', report)
    write_json(args.output / 'centroid_model.json', dict(features=FEATURES, means=means,
               scales=scales, centroids=centroids, majority=majority,
               selected=selected, preprocessing=json.loads((args.output / 'audit.json').read_text())))
    print(json.dumps({key: value for key, value in report.items() if key not in ('validation', 'test')}, indent=2))
    print('Validation macro F1:', {name: round(value['macro_f1'], 4) for name, value in validation.items()})
    print('Test accuracy:', round(test['accuracy'], 4), 'test macro F1:', round(test['macro_f1'], 4))


if __name__ == '__main__':
    main()
