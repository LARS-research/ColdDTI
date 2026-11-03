"""Generate cluster groups with the recovered single-linkage protocol."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .splits import cluster_splits, align_and_filter, audit

ROOT = Path(__file__).resolve().parents[1]


def single_linkage(features, metric, threshold, jobs=1):
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    from sklearn.metrics import pairwise_distances
    if len(features.shape) != 2 or features.shape[0] < 1:
        raise ValueError('Clustering needs at least one entity')
    if features.shape[0] == 1:
        return np.ones(1, dtype=int)
    distances = pairwise_distances(features, metric=metric, n_jobs=jobs)
    condensed = squareform(distances, checks=False)
    del distances
    return fcluster(linkage(condensed, method='single'), t=threshold, criterion='distance')


def generate(drugs, proteins, drug_threshold=.5, protein_threshold=.85, jobs=1):
    from rdkit import Chem, DataStructs
    from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
    from sklearn.feature_extraction.text import TfidfVectorizer
    generator = GetMorganGenerator(radius=2, fpSize=2048)
    fingerprints = []
    for row in drugs.itertuples():
        mol = Chem.MolFromSmiles(row.smiles)
        if mol is None:
            raise ValueError('Invalid SMILES for smiles_nid=' + str(row.smiles_nid))
        array = np.zeros(2048, dtype=bool)
        DataStructs.ConvertToNumpyArray(generator.GetFingerprint(mol), array)
        fingerprints.append(array)
    result_drugs = drugs.copy()
    result_drugs['drug_cluster'] = single_linkage(np.stack(fingerprints), 'jaccard', drug_threshold, jobs)
    vectors = TfidfVectorizer(analyzer='char', ngram_range=(3, 3), lowercase=False).fit_transform(proteins.protein)
    result_proteins = proteins.copy()
    result_proteins['target_cluster'] = single_linkage(vectors, 'cosine', protein_threshold, jobs)
    return result_drugs, result_proteins


def compare_groups(current, reference, identity, column):
    from sklearn.metrics import adjusted_rand_score
    if current[identity].duplicated().any() or reference[identity].duplicated().any():
        raise ValueError('Cluster comparison requires one row per entity identity')
    pairs = current.merge(reference[[identity, column]], on=identity, suffixes=('_new', '_saved'))
    left, right = pairs[column + '_new'], pairs[column + '_saved']
    same_partition = (pairs.groupby(column + '_new')[column + '_saved'].nunique().le(1).all()
                      and pairs.groupby(column + '_saved')[column + '_new'].nunique().le(1).all())
    return dict(shared_entities=len(pairs), generated_entities=len(current), reference_entities=len(reference),
                same_partition_on_shared_entities=bool(same_partition) if len(pairs) else False,
                adjusted_rand_score=float(adjusted_rand_score(left, right)) if len(pairs) else None)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=('drugbank', 'bindingdb', 'biosnap', 'human'), default='drugbank')
    parser.add_argument('--data-root', type=Path, default=ROOT / 'data')
    parser.add_argument('--output-dir', type=Path, required=True, help='Separate output; existing benchmark splits are preserved')
    parser.add_argument('--drug-distance', type=float, default=.5)
    parser.add_argument('--protein-distance', type=float, default=.85)
    parser.add_argument('--seed', type=int, default=57)
    parser.add_argument('--jobs', type=int, default=1)
    parser.add_argument('--compare-groups', type=Path, help='Directory containing drugs.csv and proteins.csv')
    args = parser.parse_args(argv)
    source = args.data_root / args.dataset
    if args.output_dir.resolve() == source.resolve() or source.resolve() in args.output_dir.resolve().parents:
        parser.error('Choose an output directory outside the prepared dataset')
    if not 0 <= args.drug_distance <= 1 or not 0 <= args.protein_distance <= 2:
        parser.error('Drug distance must be in [0,1], protein cosine distance in [0,2]')
    observations = pd.read_csv(source / 'observations.csv', dtype={'protein_id': str})
    drugs = pd.read_csv(source / 'smiles.csv')
    proteins = pd.read_csv(source / 'source_proteins.csv', dtype={'protein_id': str})
    drugs = drugs.loc[drugs.smiles_nid.isin(observations.smiles_nid)].sort_values('smiles_nid').reset_index(drop=True)
    proteins = proteins.loc[proteins.protein_id.isin(observations.protein_id), ['protein_id', 'protein']].drop_duplicates('protein_id')
    if set(observations.protein_id) != set(proteins.protein_id):
        raise ValueError('Source sequences are missing for observed proteins')
    d, p = generate(drugs, proteins, args.drug_distance, args.protein_distance, args.jobs)
    config = dict(method='single linkage', drug=dict(fingerprint='Morgan ECFP4', radius=2, bits=2048,
        distance='1 - Tanimoto (binary Jaccard)', threshold=args.drug_distance),
        protein=dict(features='character 3-mer TF-IDF', distance='cosine', threshold=args.protein_distance),
        split_seed=args.seed, split_assignment='prepared pipeline 80/10/10 clusters; retain same-partition pairs')
    if args.compare_groups:
        config['comparison'] = dict(
            drug=compare_groups(d, pd.read_csv(args.compare_groups / 'drugs.csv'), 'smiles_nid', 'drug_cluster'),
            protein=compare_groups(p, pd.read_csv(args.compare_groups / 'proteins.csv', dtype={'protein_id': str}), 'protein_id', 'target_cluster'))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'cluster_groups').mkdir(exist_ok=True)
    d[['smiles_nid', 'drug_cluster']].to_csv(args.output_dir / 'cluster_groups/drugs.csv', index=False)
    p[['protein_id', 'target_cluster']].to_csv(args.output_dir / 'cluster_groups/proteins.csv', index=False)
    tagged = pd.read_csv(source / 'protein.csv', dtype={'protein_id': str})
    protocol = 'cluster' if args.dataset == 'drugbank' else 'cluster_start'
    splits = align_and_filter(cluster_splits(observations, d[['smiles_nid', 'drug_cluster']], p[['protein_id', 'target_cluster']], args.seed), tagged)
    # Very strict thresholds may legitimately produce an empty evaluation set.
    # Save groups and statistics even then; only nonempty protocols pass audit.
    config['samples'] = {part: len(frame) for part, frame in splits.items()}
    config['audit_passed'] = False
    if all(len(frame) for frame in splits.values()):
        audit(splits, protocol, tagged)
        config['audit_passed'] = True
    directory = args.output_dir / protocol
    directory.mkdir(exist_ok=True)
    for part, frame in splits.items():
        frame.to_csv(directory / (part + '.csv'), index=False)
    (args.output_dir / 'clustering.json').write_text(json.dumps(config, indent=2) + '\n')
    print(json.dumps(config, indent=2))


if __name__ == '__main__':
    main()
