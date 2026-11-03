"""Check every prepared split and print concise dataset statistics."""
import argparse
import json
from pathlib import Path

import pandas as pd

from .splits import PARTS, audit


def audit_root(root, datasets):
    reports = []
    for dataset in datasets:
        directory = Path(root) / dataset
        proteins = pd.read_csv(directory / 'protein.csv', dtype={'protein_id': str})
        config_file = directory / 'preprocessing.json'
        config = json.loads(config_file.read_text()) if config_file.exists() else {}
        cutoff = config.get('temporal_protocol', {}).get('cutoff_year', 2010)
        for training in sorted(directory.glob('*/train.csv')):
            protocol = training.parent.name
            splits = {part: pd.read_csv(training.parent / (part + '.csv'), dtype={'protein_id': str})
                      for part in PARTS}
            audit(splits, protocol, proteins, cutoff)
            for part, frame in splits.items():
                reports.append(dict(dataset=dataset, protocol=protocol, split=part, samples=len(frame),
                                    drugs=frame.smiles_nid.nunique(), proteins=frame.protein_id.nunique(),
                                    positive=int(frame.Y.sum()), negative=int(frame.Y.eq(0).sum())))
    return pd.DataFrame(reports)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=Path(__file__).resolve().parents[1] / 'data')
    parser.add_argument('--dataset', nargs='+', default=['drugbank', 'bindingdb', 'biosnap', 'human'])
    args = parser.parse_args()
    report = audit_root(args.data_root, args.dataset)
    report.to_csv(args.data_root / 'split_statistics.csv', index=False)
    print(report.to_string(index=False))
    print('PASS: all pairs, held-out identities, cluster groups and protein indices checked.')


if __name__ == '__main__':
    main()
