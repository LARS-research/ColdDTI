"""Seeded basic, cluster-held-out and entity-date split construction."""
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

PARTS = ('train', 'val', 'test')
BASIC = ('random', 'cold_drug', 'cold_protein', 'cold_pair')
COLUMNS = ['smiles_nid', 'protein_nid', 'protein_id', 'Y']
PAIR = ['smiles_nid', 'protein_id']


def partition(values, seed=57):
    values = np.asarray(list(values))
    values = values[np.random.default_rng(seed).permutation(len(values))]
    a, b = int(len(values) * .8), int(len(values) * .8) + int(len(values) * .1)
    return dict(zip(PARTS, (values[:a], values[a:b], values[b:])))


def clean_pairs(frame):
    frame = frame.copy()
    if frame[PAIR + ['Y']].isna().any().any() or not frame.Y.isin([0, 1]).all():
        raise ValueError('Interactions need complete identities and binary labels')
    frame['protein_id'] = frame.protein_id.astype(str)
    frame = frame.astype({'smiles_nid': int, 'Y': int})
    conflicts = frame.groupby(PAIR, sort=False).Y.transform('nunique').gt(1)
    stats = dict(source_rows=len(frame), conflicting_rows=int(conflicts.sum()),
                 conflicting_pairs=len(frame.loc[conflicts, PAIR].drop_duplicates()))
    kept = frame.loc[~conflicts].drop_duplicates(PAIR).reset_index(drop=True)
    stats.update(duplicate_rows_removed=len(frame) - int(conflicts.sum()) - len(kept),
                 clean_observations=len(kept))
    return kept, stats


def load_raw(directory, dataset):
    directory = Path(directory)
    sources = {'bindingdb': 'bindingDB_mapped.csv', 'biosnap': 'biosnap_mapped.csv'}
    filename = sources.get(dataset, 'fulldata.csv')
    raw = pd.read_csv(directory / filename).rename(columns={'SMILES': 'smiles', 'Protein': 'protein'})
    drugs = pd.read_csv(directory / 'smiles.csv')
    protein_file = ('protein_with_timestamps.csv' if dataset == 'drugbank' and
                    (directory / 'protein_with_timestamps.csv').exists() else 'protein.csv')
    proteins = pd.read_csv(directory / protein_file, dtype={'protein_id': str})
    mapping = proteins.drop_duplicates('protein', keep='last').set_index('protein').protein_id
    observations = pd.DataFrame({'smiles_nid': raw.smiles.map(drugs.set_index('smiles').smiles_nid),
                                 'protein_id': raw.protein.map(mapping), 'Y': raw.Y})
    observations, stats = clean_pairs(observations)
    stats['source_file'] = filename
    return observations, proteins, drugs, stats


def basic_splits(observations, protocol, seed=57):
    if protocol == 'random':
        return {part: observations.iloc[indices].copy()
                for part, indices in partition(range(len(observations)), seed).items()}
    drugs = partition(observations.smiles_nid.unique(), seed)
    proteins = partition(observations.protein_id.unique(), seed)
    result = {}
    for part in PARTS:
        mask = pd.Series(True, index=observations.index)
        if protocol in ('cold_drug', 'cold_pair'):
            mask &= observations.smiles_nid.isin(drugs[part])
        if protocol in ('cold_protein', 'cold_pair'):
            mask &= observations.protein_id.isin(proteins[part])
        result[part] = observations.loc[mask].copy()
    return result


def merge_alias_clusters(frame, identity, cluster):
    """Unify clusters connected by the same actual entity identity."""
    frame = frame[[identity, cluster]].dropna().copy()
    parent = {value: value for value in frame[cluster].unique()}

    def root(value):
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    for _, group in frame.groupby(identity, sort=False):
        values = group[cluster].tolist()
        for value in values[1:]:
            a, b = root(values[0]), root(value)
            parent[max(a, b)] = min(a, b)
    frame[cluster] = frame[cluster].map(root)
    return frame.drop_duplicates(identity).reset_index(drop=True)


def normalize_cluster_inputs(directory, proteins, drugs):
    directory = Path(directory)
    drug = pd.read_csv(directory / 'drug_clusters.csv')
    target = pd.read_csv(directory / 'protein_clusters.csv')
    # The old numeric indices are not the current feature indices.
    drug['smiles_nid'] = drug.smiles.map(drugs.set_index('smiles').smiles_nid)
    target['protein_id'] = target.protein.map(
        proteins.drop_duplicates('protein', keep='last').set_index('protein').protein_id)
    return (merge_alias_clusters(drug, 'smiles_nid', 'drug_cluster'),
            merge_alias_clusters(target, 'protein_id', 'target_cluster'))


def cluster_splits(observations, drugs, proteins, seed=57):
    drugs, proteins = drugs.copy(), proteins.copy()
    for frame, column in ((drugs, 'drug_cluster'), (proteins, 'target_cluster')):
        assignments = {group: part for part, groups in partition(sorted(frame[column].unique()), seed).items()
                       for group in groups}
        frame['partition'] = frame[column].map(assignments)
    rows = observations.merge(drugs, on='smiles_nid', how='left').merge(proteins, on='protein_id',
                                                                     how='left', suffixes=('_drug', '_protein'))
    result = {part: rows.loc[rows.partition_drug.eq(part) & rows.partition_protein.eq(part)].copy()
              for part in PARTS}
    return result


def temporal_splits(observations, drugs, proteins, seed=57, cutoff=2010, val_fraction=.2):
    if not 0 < val_fraction < 1:
        raise ValueError('Temporal validation fraction must lie between zero and one')
    rows = observations.merge(drugs, on='smiles_nid', how='left').merge(proteins, on='protein_id', how='left')
    known = rows.drug_year.notna() & rows.protein_year.notna()
    historical = rows.loc[known & rows.drug_year.le(cutoff) & rows.protein_year.le(cutoff)]
    future = rows.loc[known & (rows.drug_year.gt(cutoff) | rows.protein_year.gt(cutoff))].copy()
    indices = np.random.default_rng(seed).permutation(len(historical))
    n_val = int(len(historical) * val_fraction)
    return dict(train=historical.iloc[indices[n_val:]].copy(),
                val=historical.iloc[indices[:n_val]].copy(), test=future)


def align_and_filter(splits, proteins):
    mapping = pd.Series(proteins.index, index=proteins.protein_id)
    result = {}
    for part, frame in splits.items():
        rows = frame.loc[frame.protein_id.isin(mapping.index)].copy()
        rows['protein_nid'] = rows.protein_id.map(mapping).astype(int)
        extras = [column for column in ('drug_cluster', 'target_cluster', 'drug_year', 'protein_year')
                  if column in rows]
        result[part] = rows[COLUMNS + extras].reset_index(drop=True)
    return result


def audit(splits, protocol, proteins, cutoff=2010):
    mapping = pd.Series(proteins.protein_id.values, index=proteins.index)
    for part, frame in splits.items():
        if frame.empty:
            raise ValueError(f'{protocol}/{part}: empty split')
        if frame.duplicated(PAIR).any() or not frame.Y.isin([0, 1]).all():
            raise ValueError(f'{protocol}/{part}: duplicate pair or invalid label')
        if not frame.protein_nid.map(mapping).eq(frame.protein_id).all():
            raise ValueError(f'{protocol}/{part}: protein index mismatch')
        if protocol == 'real_timeline':
            if frame[['drug_year', 'protein_year']].isna().any().any():
                raise ValueError('Temporal split contains unknown entity dates')
            valid = ((frame.drug_year.le(cutoff) & frame.protein_year.le(cutoff))
                     if part in ('train', 'val') else
                     (frame.drug_year.gt(cutoff) | frame.protein_year.gt(cutoff)))
            if not valid.all():
                raise ValueError(f'{protocol}/{part}: entity date violates cutoff {cutoff}')
    for a, b in combinations(PARTS, 2):
        left, right = splits[a], splits[b]
        if set(zip(left.smiles_nid, left.protein_id)) & set(zip(right.smiles_nid, right.protein_id)):
            raise ValueError(f'{protocol}: {a}/{b} pair overlap')
        keys = []
        if protocol in ('cold_drug', 'cold_pair', 'cluster', 'cluster_start'):
            keys.append('smiles_nid')
        if protocol in ('cold_protein', 'cold_pair', 'cluster', 'cluster_start'):
            keys.append('protein_id')
        if protocol in ('cluster', 'cluster_start'):
            keys += ['drug_cluster', 'target_cluster']
        for key in keys:
            if key in left and set(left[key]) & set(right[key]):
                raise ValueError(f'{protocol}: {a}/{b} overlapping {key}')
    return True
