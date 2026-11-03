"""Summarize retained/excluded proteins and interactions at structure filtering."""
import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def summarize(directory, dataset):
    observations = pd.read_csv(directory / 'observations.csv', dtype={'protein_id': str})
    proteins = pd.read_csv(directory / 'source_proteins.csv', dtype={'protein_id': str}).drop_duplicates('protein_id')
    coverage = pd.read_csv(directory / 'structure_coverage.csv', dtype={'protein_id': str})
    counts = observations.groupby('protein_id').Y.agg(interactions='size', positive='sum').reset_index()
    counts['negative'] = counts.interactions - counts.positive
    table = coverage.merge(proteins[['protein_id', 'protein']], on='protein_id', validate='one_to_one').merge(counts, on='protein_id', validate='one_to_one')
    if len(table) != len(coverage) or set(observations.protein_id) != set(table.protein_id):
        raise ValueError('Coverage, source proteins and observations must describe the same candidates')
    table['sequence_length'] = table.protein.str.replace(r'\s+', '', regex=True).str.len()
    table['group'] = table.status.map(lambda value: 'retained' if value == 'ok' else 'excluded')
    rows = []
    for group, frame in table.groupby('group'):
        rows.append(dict(dataset=dataset, group=group, proteins=len(frame), interactions=int(frame.interactions.sum()),
            positive=int(frame.positive.sum()), negative=int(frame.negative.sum()),
            positive_fraction=float(frame.positive.sum() / frame.interactions.sum()),
            sequence_length_mean=float(frame.sequence_length.mean()), sequence_length_median=float(frame.sequence_length.median()),
            sequence_length_q25=float(frame.sequence_length.quantile(.25)), sequence_length_q75=float(frame.sequence_length.quantile(.75)),
            interactions_per_protein_mean=float(frame.interactions.mean()), interactions_per_protein_median=float(frame.interactions.median())))
    table.insert(0, 'dataset', dataset)
    return table.drop(columns='protein'), rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=ROOT / 'data')
    parser.add_argument('--dataset', nargs='+', choices=('drugbank', 'bindingdb', 'biosnap', 'human'), default=['drugbank', 'bindingdb', 'biosnap', 'human'])
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'output/structure_coverage')
    args = parser.parse_args(argv)
    tables, summaries = [], []
    for dataset in args.dataset:
        table, rows = summarize(args.data_root / dataset, dataset)
        tables.append(table)
        summaries.extend(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = pd.DataFrame(summaries)
    report.to_csv(args.output_dir / 'retained_excluded_summary.csv', index=False)
    pd.concat(tables, ignore_index=True).to_csv(args.output_dir / 'protein_selection.csv', index=False)
    print(report.to_string(index=False))


if __name__ == '__main__':
    main()
