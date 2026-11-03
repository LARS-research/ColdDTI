"""Recovered BindingDB raw-affinity preparation and benchmark sequence mapping."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

AFFINITIES = ('Ki (nM)', 'Kd (nM)', 'IC50 (nM)', 'EC50 (nM)')
RAW_COLUMNS = ('Ligand SMILES', 'UniProt (SwissProt) Primary ID of Target Chain 1',
               'BindingDB Target Chain Sequence 1')


def map_benchmark(input_csv, protein_csv, smiles_csv, output_dir):
    """Map by exact sequence/SMILES; supplied numeric drug identities are retained."""
    frame = pd.read_csv(input_csv).rename(columns={'smiles': 'SMILES', 'protein': 'Protein'})
    proteins = pd.read_csv(protein_csv, dtype={'protein_id': str})
    drugs = pd.read_csv(smiles_csv).rename(columns={'SMILES': 'smiles'})
    if not {'SMILES', 'Protein', 'Y'}.issubset(frame.columns):
        raise ValueError('Benchmark input needs SMILES, Protein, Y')
    if not frame.Y.isin([0, 1]).all():
        raise ValueError('Benchmark labels must be binary')
    proteins = proteins.copy()
    proteins['protein'] = proteins.protein.str.strip()
    proteins['protein_id'] = proteins.protein_id.str.strip()
    proteins['source_protein_nid'] = proteins.index
    # Match the recovered mapping script's last sequence-to-identity assignment.
    mapping = proteins.drop_duplicates('protein', keep='last').set_index('protein')
    frame['Protein'] = frame.Protein.str.strip()
    frame['SMILES'] = frame.SMILES.str.strip()
    frame['protein_id'] = frame.Protein.map(mapping.protein_id)
    frame['protein_nid'] = frame.Protein.map(mapping.source_protein_nid)
    frame['smiles_nid'] = frame.SMILES.map(drugs.set_index('smiles').smiles_nid)
    valid = frame[['protein_id', 'protein_nid', 'smiles_nid']].notna().all(axis=1)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rejected = frame.loc[~valid].copy()
    rejected['reason'] = np.where(rejected.protein_id.isna(), 'unmapped protein sequence', 'unmapped SMILES')
    rejected.to_csv(output / 'unmapped.csv', index=False)
    retained = frame.loc[valid].astype({'smiles_nid': int, 'protein_nid': int, 'Y': int})
    order = ['smiles_nid', 'protein_nid', 'protein_id', 'SMILES', 'Protein', 'Y']
    retained[order + [c for c in frame.columns if c not in order]].to_csv(output / 'bindingDB_mapped.csv', index=False)
    stats = dict(input_rows=len(frame), mapped_rows=len(retained), excluded_rows=len(rejected),
                 mapping='exact stripped sequence and SMILES; supplied smiles_nid retained',
                 excluded_by_reason=rejected.reason.value_counts().to_dict())
    (output / 'mapping.json').write_text(json.dumps(stats, indent=2) + '\n')
    return stats


def prepare_raw(input_tsv, output_dir, positive=6., negative=5., censored='exclude', chunksize=50000):
    """Ki -> Kd -> IC50 -> EC50 priority, p=-log10(nM*1e-9), canonical SMILES."""
    from rdkit import Chem
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    temporary = output / '.affinity_rows.csv'
    counts, smiles_cache, all_smiles, protein_records = Counter(), {}, set(), {}
    header_written = False
    for frame in pd.read_csv(input_tsv, sep='\t', usecols=list(RAW_COLUMNS + AFFINITIES), chunksize=chunksize, low_memory=False):
        counts['raw_rows'] += len(frame)
        frame = frame.dropna(subset=list(RAW_COLUMNS)).copy()
        for column in RAW_COLUMNS:
            frame[column] = frame[column].str.strip()
            frame = frame.loc[frame[column].ne('')]
        counts['complete_identity_rows'] += len(frame)
        affinities = []
        for column in AFFINITIES:
            values = frame[column].astype(str)
            numeric = pd.to_numeric(values.str.extract(r'([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)', expand=False), errors='coerce')
            if censored == 'exclude':
                numeric = numeric.mask(values.str.contains(r'[<>~≤≥]', regex=True))
            affinities.append(numeric.where(numeric.gt(0)))
        selected = pd.concat(affinities, axis=1).bfill(axis=1).iloc[:, 0]
        p_affinity = -np.log10(selected * 1e-9)
        counts['usable_affinity_rows'] += int(selected.notna().sum())
        labels = pd.Series(np.nan, index=frame.index)
        labels.loc[p_affinity.ge(positive)] = 1
        labels.loc[p_affinity.le(negative)] = 0
        frame['Y'] = labels
        frame = frame.loc[frame.Y.notna()].copy()
        counts['binary_affinity_rows'] += len(frame)

        def canonical(value):
            value = value.strip().strip('"').split('|')[0].strip()
            if value not in smiles_cache:
                mol = Chem.MolFromSmiles(value)
                smiles_cache[value] = Chem.MolToSmiles(mol) if mol is not None else None
            return smiles_cache[value]

        frame['smiles'] = frame[RAW_COLUMNS[0]].map(canonical)
        frame = frame.loc[frame.smiles.notna()].copy()
        counts['valid_smiles_rows'] += len(frame)
        result = frame.rename(columns={RAW_COLUMNS[1]: 'protein_id', RAW_COLUMNS[2]: 'protein'})[
            ['protein_id', 'smiles', 'protein', 'Y']].astype({'Y': int})
        all_smiles.update(result.smiles)
        for row in result[['protein_id', 'protein']].drop_duplicates().itertuples(index=False):
            protein_records[(row.protein_id, row.protein)] = None
        result.to_csv(temporary, mode='a' if header_written else 'w', header=not header_written, index=False)
        header_written = True
    if not header_written or not all_smiles:
        raise ValueError('No rows survived affinity preparation')
    drugs = pd.DataFrame({'smiles': sorted(all_smiles)})
    drugs.insert(0, 'smiles_nid', range(len(drugs)))
    mapping = drugs.set_index('smiles').smiles_nid
    first = True
    for frame in pd.read_csv(temporary, chunksize=chunksize, dtype={'protein_id': str}):
        frame.insert(0, 'smiles_nid', frame.smiles.map(mapping).astype(int))
        frame.to_csv(output / 'fulldata.csv', mode='w' if first else 'a', header=first, index=False)
        first = False
    temporary.unlink()
    drugs.to_csv(output / 'smiles.csv', index=False)
    pd.DataFrame(protein_records.keys(), columns=['protein_id', 'protein']).to_csv(output / 'protein.csv', index=False)
    stats = dict(stages=dict(counts), affinity_priority=list(AFFINITIES), positive_p=positive,
                 negative_p=negative, censored_policy=censored,
                 qualifiers='exclude inequalities/approximation' if censored == 'exclude' else 'source legacy numeric extraction',
                 pair_deduplication='not performed here; handled by prepared benchmark pipeline')
    (output / 'raw_preprocessing.json').write_text(json.dumps(stats, indent=2) + '\n')
    return stats


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    mapping = commands.add_parser('map', help='Map an existing labeled benchmark by sequence and SMILES')
    mapping.add_argument('--input-csv', type=Path, required=True)
    mapping.add_argument('--protein-csv', type=Path, required=True)
    mapping.add_argument('--smiles-csv', type=Path, required=True)
    raw = commands.add_parser('raw', help='Prepare a raw BindingDB affinity TSV independently')
    raw.add_argument('--input-tsv', type=Path, required=True)
    raw.add_argument('--positive-p', type=float, default=6.)
    raw.add_argument('--negative-p', type=float, default=5.)
    raw.add_argument('--censored-policy', choices=('exclude', 'legacy_numeric'), default='exclude')
    for command in (mapping, raw):
        command.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    # Recovered tools write into a fresh directory and never overwrite source tables.
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error('--output-dir must be new or empty')
    if args.command == 'map':
        result = map_benchmark(args.input_csv, args.protein_csv, args.smiles_csv, args.output_dir)
    else:
        if args.positive_p <= args.negative_p:
            parser.error('--positive-p must exceed --negative-p')
        result = prepare_raw(args.input_tsv, args.output_dir, args.positive_p, args.negative_p, args.censored_policy)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
