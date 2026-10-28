"""Predict activities from the feature windows produced by pipeline.py."""

import argparse
from pathlib import Path


def main():
    import joblib
    import pandas as pd
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='outputs/baseline.joblib')
    parser.add_argument('--windows', default='outputs/windows.csv')
    parser.add_argument('--output', default='outputs/predictions.csv')
    args = parser.parse_args()
    artifact = joblib.load(args.model)
    frame = pd.read_csv(args.windows)
    missing = set(artifact['features']) - set(frame.columns)
    if missing:
        raise ValueError(f'Missing feature columns: {sorted(missing)}')
    features = frame[artifact['features']]
    if features.isna().any().any() or not features.map(lambda x: float('-inf') < x < float('inf')).all().all():
        raise ValueError('Features must contain finite numeric values')
    result = frame[[column for column in ('user_id', 'interval_id', 'session', 'start', 'split')
                    if column in frame]].copy()
    result['predicted_activity_type_id'] = artifact['model'].predict(features)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(destination, index=False)
    print(f'Saved {len(result)} predictions to {destination}')


if __name__ == '__main__':
    main()
