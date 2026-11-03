"""Export checkpoint interaction maps and residue annotations for selected test pairs."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ablations.boundaries import secondary_units
from model.checkpoint import load_checkpoint
from model.segments import marker_spans

ROOT = Path(__file__).resolve().parent


def source_positions(recovered, canonical):
    """Map recovered residues to a supplied source sequence, retaining exact matches."""
    if recovered == canonical:
        return list(range(1, len(recovered) + 1))
    from Bio.Align import PairwiseAligner
    aligner = PairwiseAligner(mode='global', match_score=2, mismatch_score=-1,
                             open_gap_score=-4, extend_gap_score=-.1)
    alignment = aligner.align(canonical, recovered)[0]
    positions = [None] * len(recovered)
    for (a, b), (c, d) in zip(*alignment.aligned):
        for source, target in zip(range(a, b), range(c, d)):
            if canonical[source] == recovered[target]:
                positions[target] = int(source + 1)
    return positions


def protein_units(tokens, model, protein_key):
    units = {'p': [[i] for i in range(len(tokens))], 'q': [list(range(len(tokens)))]}
    for level, prefix in (('s', 'secondary'), ('t', 'tertiary')):
        start, end = getattr(model, prefix + '_start_id'), getattr(model, prefix + '_end_id')
        spans = marker_spans(tokens, start, end)
        units[level] = [list(range(left - 1, right + 1)) for left, right in spans] or [list(range(len(tokens)))]
    if model.variant.endswith('_secondary') and model.variant != 'wo_secondary':
        units['s'] = secondary_units(tokens, model.variant.split('_')[0], seed=model.boundary_seed,
            protein_key=protein_key, secondary_start=model.secondary_start_id, secondary_end=model.secondary_end_id,
            tertiary_start=model.tertiary_start_id, tertiary_end=model.tertiary_end_id) or [list(range(len(tokens)))]
    return units


def select_cases(frame, args, directory):
    selected = frame.copy()
    if args.predictions:
        predictions = pd.read_csv(args.predictions, dtype={'protein_id': str})
        score = 'score' if 'score' in predictions else 'y_score'
        keys = ['smiles_nid', 'protein_id', 'Y']
        joined = predictions[keys + [score]].merge(frame[keys], on=keys, how='left', indicator=True, validate='one_to_one')
        if joined['_merge'].ne('both').any() or not np.isfinite(joined[score]).all():
            raise ValueError('Predictions must refer to unique current test rows with finite scores')
        selected = frame.merge(predictions[keys + [score]].rename(columns={score: 'selected_score'}), on=keys, validate='one_to_one')
        selected = selected.sort_values('selected_score', ascending=False, kind='stable')
    if args.drugbank_id:
        mapping = pd.read_csv(directory / 'drug_identifiers.csv')
        ids = mapping.loc[mapping.drugbank_id.eq(args.drugbank_id), 'smiles_nid'].unique()
        if len(ids) != 1:
            raise ValueError('DrugBank ID must map to exactly one actual SMILES identity')
        selected = selected.loc[selected.smiles_nid.eq(ids[0])]
    if args.drug_nid is not None:
        selected = selected.loc[selected.smiles_nid.eq(args.drug_nid)]
    if args.protein_id:
        selected = selected.loc[selected.protein_id.eq(args.protein_id)]
    if selected.empty:
        raise ValueError('No matching pair in the selected checkpoint test partition')
    return selected.head(args.top_k)


def row_softmax(values, temperature):
    shifted = values / temperature
    shifted -= shifted.max(axis=1, keepdims=True)
    exponential = np.exp(shifted)
    return exponential / exponential.sum(axis=1, keepdims=True)


def plot_maps(raw, units, drug_tokens, labels, positions, regions, directory, temperature, top_tokens):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = [name for name in ('lp', 'ls') if name in raw]
    if not names:
        return
    maps = {name: row_softmax(raw[name], temperature) for name in names}
    special = {'<s>', '</s>', '[PAD]', '[CLS]', '[SEP]', '[MASK]', '[UNK]'}
    candidates = [i for i, token in enumerate(drug_tokens) if labels.get(str(token), str(token)) not in special]
    if not candidates:
        candidates = list(range(len(drug_tokens)))
    strength = sum(value.max(axis=1) for value in maps.values())
    rows = sorted(candidates, key=lambda i: (-strength[i], i))[:top_tokens]
    vmax = max(float(value[rows].max()) for value in maps.values())
    fig, axes = plt.subplots(len(names), 1, figsize=(15, max(3, len(rows) * .3) * len(names)), squeeze=False)
    for ax, name in zip(axes[:, 0], names):
        value = maps[name][rows]
        heatmap = ax.imshow(value, aspect='auto', cmap='viridis', vmin=0, vmax=vmax, interpolation='nearest')
        ax.set_yticks(range(len(rows)), [str(i) + ': ' + labels.get(str(drug_tokens[i]), str(drug_tokens[i])) for i in rows])
        ax.set_xlabel('Protein token index' if name == 'lp' else 'Secondary unit index')
        ax.set_ylabel('Drug token')
        ax.set_title(name + ': row-softmax interaction scores (visualization temperature=' + str(temperature) + ')')
        for column, indices in enumerate(units[name[1]]):
            canonical = [positions[i] for i in indices if positions[i] is not None]
            if any(any(start <= position <= end for position in canonical) for start, end in regions):
                ax.axvspan(column - .5, column + .5, ymin=.97, ymax=1, color='red', alpha=.6)
        fig.colorbar(heatmap, ax=ax, label='Visualization row-softmax weight')
    fig.tight_layout()
    fig.savefig(directory / 'interaction_maps.png', dpi=180)
    fig.savefig(directory / 'interaction_maps.pdf')
    plt.close(fig)


def export_case(row, model, features, config, args, tables, token_labels):
    protein_table, source_table, regions_table, drugs = tables
    protein_id, protein_nid, drug_nid = str(row.protein_id), int(row.protein_nid), int(row.smiles_nid)
    if str(protein_table.iloc[protein_nid].protein_id) != protein_id:
        raise ValueError('Test protein_nid and protein_id disagree with the model input table')
    values = features.pair(drug_nid, protein_nid)
    with torch.no_grad():
        logits, explanation = model(*(value.to(args.device) for value in values), protein_key=protein_nid, return_maps=True)
    probability = float(logits.softmax(dim=-1)[0, 1])
    if hasattr(row, 'selected_score') and abs(probability - row.selected_score) > 1e-5:
        raise ValueError('Supplied prediction scores differ from this checkpoint and feature snapshot')
    raw = {key: value[0].detach().cpu().numpy() for key, value in explanation['interaction_maps'].items()}
    tokens = values[1].reshape(-1).tolist()
    labels = dict(token_labels['prot_resize'])
    for prefix in ('secondary', 'tertiary'):
        for end in ('start', 'end'):
            labels[str(getattr(model, prefix + '_' + end + '_id'))] = '[' + prefix + '_' + end + ']'
    residue_indices = [i for i, token in enumerate(tokens) if 5 <= token <= 29]
    recovered = ''.join(labels[str(tokens[i])] for i in residue_indices)
    source = source_table.loc[source_table.protein_id.eq(protein_id), 'protein']
    if source.empty:
        raise ValueError('Source protein sequence is unavailable for residue alignment')
    canonical = str(source.iloc[-1]).replace(' ', '')
    correspondence = source_positions(recovered, canonical)
    positions = [None] * len(tokens)
    recovered_indices = [None] * len(tokens)
    for number, index in enumerate(residue_indices):
        positions[index], recovered_indices[index] = correspondence[number], number + 1
    units = protein_units(tokens, model, protein_nid)
    unit_rows = []
    for level in explanation['protein_scores']:
        score = explanation['protein_scores'][level][0]
        weights = (score / model.sqrt_dk).softmax(dim=-1).detach().cpu().numpy()
        for i, indices in enumerate(units[level]):
            source_ids = [positions[index] for index in indices if positions[index] is not None]
            unit_rows.append(dict(level=level, unit_index=i, token_start=min(indices), token_end=max(indices),
                                  source_positions=';'.join(map(str, source_ids)), pooling_weight=float(weights[i])))
    token_rows = [dict(token_index=i, token_id=token, token=labels.get(str(token), str(token)),
                       recovered_position=recovered_indices[i], source_position=positions[i]) for i, token in enumerate(tokens)]
    region_rows = regions_table.loc[regions_table.protein_id.eq(protein_id)]
    if len(region_rows) and 'reference_sequence_sha256' in region_rows:
        digest = hashlib.sha256(canonical.encode()).hexdigest()
        if not region_rows.reference_sequence_sha256.eq(digest).all():
            raise ValueError('Reference-region coordinates belong to a different source sequence version')
    directory = args.output_dir / (str(drug_nid) + '_' + protein_id)
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(directory / 'interaction_maps.npz', **raw)
    pd.DataFrame(token_rows).to_csv(directory / 'protein_tokens.csv', index=False)
    pd.DataFrame(unit_rows).to_csv(directory / 'protein_units.csv', index=False)
    region_rows.to_csv(directory / 'reference_regions.csv', index=False)
    info = dict(protein_id=protein_id, protein_nid=protein_nid, smiles_nid=drug_nid,
                smiles=drugs.set_index('smiles_nid').loc[drug_nid, 'smiles'], label=int(row.Y), score=probability,
                checkpoint=str(args.ckpt), variant=model.variant, split=config['split'],
                recovered_residues=len(recovered), matched_source_residues=sum(p is not None for p in correspondence),
                reference_numbering='one-based source sequence; gaps/mismatches have no source position',
                interpretation='Descriptive model compatibility/pooling scores; not physical contacts or biological validation')
    (directory / 'case.json').write_text(json.dumps(info, indent=2) + '\n')
    if not args.no_plots:
        plot_maps(raw, units, values[0].reshape(-1).tolist(), token_labels['ChemBERTa-77M-MLM'], positions,
                  list(zip(region_rows.start, region_rows.end)), directory, args.temperature, args.top_tokens)
    return info


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ckpt', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, default=ROOT / 'data')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'output/case_study')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--predictions', type=Path, help='Rank matching test predictions by score')
    identity = parser.add_mutually_exclusive_group()
    identity.add_argument('--drug-nid', type=int)
    identity.add_argument('--drugbank-id')
    parser.add_argument('--protein-id')
    parser.add_argument('--top-k', type=int, default=5)
    parser.add_argument('--top-tokens', type=int, default=10)
    parser.add_argument('--temperature', type=float, default=.5)
    parser.add_argument('--no-plots', action='store_true')
    args = parser.parse_args(argv)
    if args.top_k < 1 or args.top_tokens < 1 or args.temperature <= 0:
        parser.error('Case counts and visualization temperature must be positive')
    if not args.predictions and args.drug_nid is None and not args.drugbank_id and not args.protein_id:
        parser.error('Specify a pair identity or --predictions for confidence-based selection')
    model, features, config = load_checkpoint(args.ckpt, args.data_root, args.device)
    directory = features.directory
    test, _ = features.read_split(directory / config['split'] / 'test.csv')
    cases = select_cases(test, args, directory)
    proteins = pd.read_csv(directory / 'protein.csv', dtype={'protein_id': str})
    source = pd.read_csv(directory / 'source_proteins.csv', dtype={'protein_id': str})
    region_path = directory / 'protein_regions.csv'
    regions = pd.read_csv(region_path, dtype={'protein_id': str}) if region_path.exists() else pd.DataFrame(columns=['protein_id', 'start', 'end'])
    drugs = pd.read_csv(directory / 'smiles.csv')
    labels = json.loads((ROOT / 'resources/token_labels.json').read_text())
    results = [export_case(row, model, features, config, args, (proteins, source, regions, drugs), labels) for row in cases.itertuples()]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(args.output_dir / 'selected_cases.csv', index=False)
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
