"""Prepare all dataset tables with one structure map and seed=57."""
import argparse
import copy
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import shutil
import tempfile

import pandas as pd
from Bio.PDB.PDBExceptions import PDBConstructionException

from .splits import (PARTS, BASIC, load_raw, basic_splits, cluster_splits, temporal_splits,
                     normalize_cluster_inputs, align_and_filter, audit)
from .structure import extract_hierarchical, process
from .artifacts import dataset_lock
from .publication import publish, recover

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ('drugbank', 'bindingdb', 'biosnap', 'human')


def prepare_structure(job):
    identifier, path, mode, ranges, chains, model_index = job
    detail = dict(protein_id=identifier, status='ok', reason='', chains=0,
                  secondary_units=0, tertiary_units=0, tertiary_source='')
    record = None
    try:
        hierarchy = extract_hierarchical(path, mode, ranges, chains=chains, model_index=model_index)
        tagged = process(hierarchy)
        detail.update(chains=len(hierarchy), secondary_units=tagged.count('[secondary_start]'),
                      tertiary_units=tagged.count('[tertiary_start]'),
                      tertiary_source=';'.join(sorted({chain['tertiary_source'] for chain in hierarchy.values()})))
        record = dict(protein_id=identifier, protein=tagged)
    except (OSError, ValueError, KeyError, IndexError, TypeError, PDBConstructionException) as error:
        detail.update(status='missing' if not path.is_file() else 'empty' if not path.stat().st_size else 'parse_failed',
                      reason=f'{type(error).__name__}: {error}')
    return record, detail


def prepare(args, dataset):
    directory = args.data_root / dataset
    directory.mkdir(parents=True, exist_ok=True)
    with dataset_lock(directory, exclusive=True):
        recover(directory)
        with tempfile.TemporaryDirectory(prefix='.preprocess-' + dataset + '-', dir=directory.parent) as temporary:
            staging_root = Path(temporary)
            staging = staging_root / dataset
            staging.mkdir()
            # Copy only small tables and metadata. Feature arrays and CIFs stay
            # in their original directories and are never copied or followed.
            for source in directory.rglob('*'):
                relative = source.relative_to(directory)
                if any(part.startswith('.') for part in relative.parts):
                    continue
                if source.is_file() and source.suffix in ('.csv', '.json'):
                    target = staging / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source, target)
            staged_args = copy.copy(args)
            staged_args.data_root = staging_root
            staged_args.cif_root = args.cif_root or args.data_root
            _prepare(staged_args, dataset)
            from .audit import audit_root
            audit_root(staging_root, [dataset])
            expected = set(json.loads((staging / 'preprocessing.json').read_text())['protocols'])
            actual = {path.parent.name for path in staging.glob('*/train.csv')}
            if actual != expected:
                raise ValueError('Every existing protocol must be regenerated before publication')
            publish(staging, directory)


def _prepare(args, dataset):
    directory = args.data_root / dataset
    directory.mkdir(parents=True, exist_ok=True)
    raw = args.raw_root / dataset if args.raw_root else None
    if raw:
        observations, source_proteins, drugs, cleaning = load_raw(raw, dataset)
        observations.to_csv(directory / 'observations.csv', index=False)
        source_proteins.to_csv(directory / 'source_proteins.csv', index=False)
        drugs.to_csv(directory / 'smiles.csv', index=False)
    else:
        observations = pd.read_csv(directory / 'observations.csv', dtype={'protein_id': str})
        source_proteins = pd.read_csv(directory / 'source_proteins.csv', dtype={'protein_id': str})
        drugs = pd.read_csv(directory / 'smiles.csv')
        cleaning = json.loads((directory / 'preprocessing.json').read_text())['source_cleaning']

    domain_file = args.tertiary_ranges or directory / 'domain_ranges.csv'
    domain_table = (pd.read_csv(domain_file, dtype={'protein_id': str, 'chain_id': str})
                    if args.tertiary_ranges or domain_file.exists() else None)
    if domain_table is not None:
        required = {'protein_id', 'chain_id', 'start', 'end'}
        if not required.issubset(domain_table.columns):
            raise ValueError('Domain CSV needs protein_id,chain_id,start,end')
        domain_table.to_csv(directory / 'domain_ranges.csv', index=False)
    cif_root = args.cif_root / dataset / 'cif' if args.cif_root else directory / 'cif'
    needed = set(observations.protein_id)
    candidates = source_proteins.loc[source_proteins.protein_id.isin(needed)].drop_duplicates('protein_id')
    jobs = []
    for row in candidates.itertuples():
        ranges = (domain_table.loc[domain_table.protein_id.eq(row.protein_id)].to_dict('records')
                  if domain_table is not None else None)
        jobs.append((row.protein_id, cif_root / (row.protein_id + '.cif'), args.tertiary_mode,
                     ranges, args.chains, args.model_index))
    records, coverage = [], []
    executor = ProcessPoolExecutor(max_workers=args.workers) if args.workers > 1 else None
    try:
        results = executor.map(prepare_structure, jobs, chunksize=4) if executor else map(prepare_structure, jobs)
        for number, (record, detail) in enumerate(results, 1):
            if record is not None:
                records.append(record)
            coverage.append(detail)
            if number % 100 == 0 or number == len(candidates):
                print(f'{dataset}: structures {number}/{len(candidates)}, accepted={len(records)}', flush=True)
    finally:
        if executor:
            executor.shutdown()
    proteins = pd.DataFrame(records, columns=['protein_id', 'protein'])
    if proteins.empty:
        raise ValueError(f'{dataset}: no valid structures for the requested tag mode')
    protocols = {name: basic_splits(observations, name, args.seed) for name in BASIC}

    cluster_root = directory / 'cluster_groups'
    upstream_clusters = raw / 'cluster_start' if raw else None
    if upstream_clusters and (upstream_clusters / 'drug_clusters.csv').exists():
        drug_groups, protein_groups = normalize_cluster_inputs(upstream_clusters, source_proteins, drugs)
        cluster_root.mkdir(exist_ok=True)
        drug_groups.to_csv(cluster_root / 'drugs.csv', index=False)
        protein_groups.to_csv(cluster_root / 'proteins.csv', index=False)
    elif (cluster_root / 'drugs.csv').exists():
        drug_groups = pd.read_csv(cluster_root / 'drugs.csv')
        protein_groups = pd.read_csv(cluster_root / 'proteins.csv', dtype={'protein_id': str})
    else:
        drug_groups = protein_groups = None
    if drug_groups is not None:
        cluster_name = 'cluster' if dataset == 'drugbank' else 'cluster_start'
        protocols[cluster_name] = cluster_splits(observations, drug_groups, protein_groups, args.seed)

    if dataset == 'drugbank':
        if raw:
            dated_drugs = pd.read_csv(raw / 'smiles_with_drugid_with_approval_time.csv')
            dated_drugs['smiles_nid'] = dated_drugs.smiles.map(drugs.set_index('smiles').smiles_nid)
            dated_drugs['drug_year'] = pd.to_numeric(dated_drugs.approval_year, errors='coerce')
            dated_drugs = dated_drugs.groupby('smiles_nid', as_index=False).drug_year.min()
            dated_proteins = source_proteins[['protein_id', 'uniprot_entry_created']].copy()
            dated_proteins['protein_year'] = pd.to_datetime(dated_proteins.uniprot_entry_created, errors='coerce').dt.year
            dated_proteins = dated_proteins.groupby('protein_id', as_index=False).protein_year.min()
            dated_drugs.to_csv(directory / 'drug_dates.csv', index=False)
            dated_proteins[['protein_id', 'protein_year']].to_csv(directory / 'protein_dates.csv', index=False)
        else:
            dated_drugs = pd.read_csv(directory / 'drug_dates.csv')
            dated_proteins = pd.read_csv(directory / 'protein_dates.csv', dtype={'protein_id': str})
        protocols['real_timeline'] = temporal_splits(observations, dated_drugs,
            dated_proteins, args.seed, args.cutoff_year, args.temporal_val_fraction)

    prepared, stats = {}, []
    for name, splits in protocols.items():
        aligned = align_and_filter(splits, proteins)
        audit(aligned, name, proteins, args.cutoff_year)
        prepared[name] = aligned
        for part, frame in aligned.items():
            stats.append(dict(dataset=dataset, protocol=name, split=part, seed=args.seed,
                              samples=len(frame), drugs=frame.smiles_nid.nunique(),
                              proteins=frame.protein_id.nunique(), positive=int(frame.Y.sum()),
                              negative=int(frame.Y.eq(0).sum()),
                              removed_unavailable_structure=len(splits[part]) - len(frame)))
    # One protein table and one map are used by EVERY split. Existing encoder
    # files are kept as historical assets but explicitly invalidated.
    (directory / 'feature_status.json').write_text(json.dumps(dict(ready=False,
        reason='Structure tags/indices were regenerated. Run get_emb.py for this data directory.'), indent=2) + '\n')
    proteins.to_csv(directory / 'protein.csv', index=False)
    pd.DataFrame(coverage).to_csv(directory / 'structure_coverage.csv', index=False)
    for name, splits in prepared.items():
        (directory / name).mkdir(exist_ok=True)
        for part, frame in splits.items():
            frame.to_csv(directory / name / (part + '.csv'), index=False)
    pd.DataFrame(stats).to_csv(directory / 'split_statistics.csv', index=False)
    config = dict(dataset=dataset, split_seed=args.seed, source_cleaning=cleaning,
                  tertiary_mode=args.tertiary_mode, domain_annotation_csv=domain_table is not None,
                  model_index=args.model_index, selected_chains=args.chains,
                  structure_rule='available, nonempty, successfully parsed, amino-acid sequence recovered',
                  numbering='mmCIF label_asym_id and label_seq_id; mapped to recovered residue positions',
                  secondary_fields=['struct_conf', 'struct_sheet_range'],
                  tertiary_domain_fields='pdbx_domain_range or explicit label-residue CSV',
                  candidates=len(candidates), retained_proteins=len(proteins),
                  structure_status_counts=pd.Series([r['status'] for r in coverage]).value_counts().to_dict(),
                  basic_ratio='80/10/10 interaction counts for random; held-out entity counts for cold',
                  basic_partition_order='partition clean source observations, then remove unavailable structures',
                  cluster_protocol='frozen upstream cluster groups, merged by actual entity identity; groups repartitioned 80/10/10',
                  cluster_upstream_method='Legacy cluster membership snapshot; original clustering provenance not verified',
                  temporal_protocol=dict(cutoff_year=args.cutoff_year, known_dates_required=True,
                    train='both years <= cutoff', test='at least one year > cutoff',
                    validation_fraction_of_historical=args.temporal_val_fraction,
                    note='entity-date split; no interaction discovery timestamps'),
                  protocols=list(prepared), audit_passed=True)
    (directory / 'preprocessing.json').write_text(json.dumps(config, indent=2) + '\n')
    print(pd.DataFrame(stats)[['protocol', 'split', 'samples', 'positive', 'negative']].to_string(index=False), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', nargs='+', choices=(*DATASETS, 'all'), default=['drugbank'])
    parser.add_argument('--data-root', type=Path, default=ROOT / 'data')
    parser.add_argument('--raw-root', type=Path, help='Optional original source tables; otherwise use the saved input snapshots')
    parser.add_argument('--cif-root', type=Path, help='Root containing DATASET/cif/; defaults to --data-root')
    parser.add_argument('--seed', type=int, default=57)
    parser.add_argument('--tertiary-mode', choices=('chain', 'domain', 'auto', 'none'), default='chain')
    parser.add_argument('--tertiary-ranges', type=Path, help='CSV: protein_id,chain_id,start,end; one-based label_seq_id')
    parser.add_argument('--chains', nargs='+', help='Optional label_asym_id selection')
    parser.add_argument('--model-index', type=int, default=0)
    parser.add_argument('--workers', type=int, default=4, help='CPU structure readers; result order stays fixed')
    parser.add_argument('--cutoff-year', type=int, default=2010)
    parser.add_argument('--temporal-val-fraction', type=float, default=.2)
    parser.add_argument('--types', nargs='+', help='Compatibility only: all split indices are always regenerated together')
    parser.add_argument('--recover', action='store_true', help='Restore an interrupted table publication without rebuilding')
    args = parser.parse_args(argv)
    for dataset in DATASETS if 'all' in args.dataset else args.dataset:
        if args.recover:
            directory = args.data_root / dataset
            with dataset_lock(directory, exclusive=True):
                restored = recover(directory)
            print(dataset + (': interrupted publication recovered' if restored else ': no interrupted publication'))
        else:
            prepare(args, dataset)


if __name__ == '__main__':
    main()
