"""Bind downstream weights to their implementation, features and split snapshot."""
import json
from pathlib import Path
import uuid

from preprocess.artifacts import atomic_json, dataset_lock, sha256

CHECKPOINT_SCHEMA = 1
IMPLEMENTATION = 'colddti-batch-first-scopes-v1'
OPTIONS = ('smile_dim', 'protein_dim', 'dim', 'boundary_seed', 'secondary_start_id',
           'secondary_end_id', 'tertiary_start_id', 'tertiary_end_id')


def make_contract(config, features):
    features.assert_current()
    if config.get('token_attention_batch_first') is not True:
        raise ValueError('Only the corrected batch_first attention implementation is supported')
    options = {key: config[key] for key in OPTIONS}
    if options['smile_dim'] != features.metadata['branches']['drug']['dimension'] or options['protein_dim'] != features.metadata['branches']['protein']['dimension']:
        raise ValueError('Model dimensions do not match the frozen feature dimensions')
    if options['smile_dim'] + options['protein_dim'] != 2 * options['dim']:
        raise ValueError('The classifier requires 2*dim == smile_dim + protein_dim')
    for name, token in features.metadata['tag_ids'].items():
        if options[name + '_id'] != token:
            raise ValueError('Configured hierarchy token IDs differ from the feature tokenizer')
    split = config['split']
    if not config.get('run_id'):
        raise ValueError('Checkpoint configuration requires a unique run identity')
    files = ['protein.csv', 'smiles.csv'] + [split + '/' + part + '.csv' for part in ('train', 'val', 'test')]
    return dict(schema_version=CHECKPOINT_SCHEMA, implementation=IMPLEMENTATION,
                token_attention_batch_first=True, dataset=config['dataset'], split=split,
                variant=config['variant'], options=options, feature_id=features.metadata['feature_id'],
                run_id=config['run_id'],
                training={key: config.get(key) for key in ('seed', 'batch_size', 'epochs', 'lr', 'weight_decay',
                    'decay_interval', 'lr_decay', 'patience', 'min_delta', 'train_samples')},
                data={name: sha256(features.directory / name) for name in files})


def save_checkpoint(path, model, contract):
    import os
    import tempfile
    import torch
    path = Path(path)
    descriptor, name = tempfile.mkstemp(prefix='.' + path.name + '-', dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, 'wb') as handle:
            torch.save(dict(state_dict=model.state_dict(), artifact_contract=contract), handle)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def checkpoint_state(path, expected, device):
    import torch
    payload = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(payload, dict) or 'state_dict' not in payload or payload.get('artifact_contract') != expected:
        raise RuntimeError('Checkpoint lacks matching embedded provenance. Unversioned/legacy weights '
                           'must be verified against their original code, data and encoders before use.')
    return payload['state_dict']


def read_config(path):
    path = Path(path)
    configuration = path.parent / 'config.json'
    if not configuration.is_file():
        raise RuntimeError('Checkpoint config.json is required; defaults cannot identify an older model')
    config = json.loads(configuration.read_text())
    contract = config.get('artifact_contract', {})
    if config.get('token_attention_batch_first') is not True or contract.get('token_attention_batch_first') is not True:
        raise RuntimeError('Checkpoint attention layout is missing or incompatible with batch_first=True')
    if contract.get('schema_version') != CHECKPOINT_SCHEMA or contract.get('implementation') != IMPLEMENTATION:
        raise RuntimeError('Checkpoint implementation version is missing or unsupported; legacy weights are not loaded automatically')
    return config


def load_checkpoint(path, data_root=None, device='cpu', dataset=None, split=None, variant=None):
    from ablations.data import FeatureStore
    from ablations.model import ColdDTIAblation
    config = read_config(path)
    for key, value in (('dataset', dataset), ('split', split), ('variant', variant)):
        if value is not None and value != config[key]:
            raise ValueError('Cannot override checkpoint ' + key + '; use its recorded experimental setting')
    root = Path(data_root) if data_root is not None else Path(config['data_root'])
    directory = root / config['dataset']
    with dataset_lock(directory):
        features = FeatureStore(directory)
        current = make_contract(config, features)
        expected = config['artifact_contract']
        if current != expected:
            changed = [key for key in current if current[key] != expected.get(key)]
            raise RuntimeError('Checkpoint does not match current artifacts: ' + ', '.join(changed))
        model = ColdDTIAblation(config['variant'], **current['options']).to(device)
        model.load_state_dict(checkpoint_state(path, expected, device), strict=True)
        features.expected_splits = {name: digest for name, digest in current['data'].items() if '/' in name}
        features.assert_current()
    return model.eval(), features, config


def register_legacy(path, data_root, reference_predictions, output_dir, device='cpu'):
    """Register corrected-layout weights only after replaying original predictions."""
    import numpy as np
    import pandas as pd
    import torch
    from torch.utils.data import DataLoader
    from ablations.data import FeatureStore, InteractionDataset, keep_variable_lengths
    from ablations.model import ColdDTIAblation
    from ablations.run import evaluate

    path, output_dir = Path(path), Path(output_dir)
    if output_dir.resolve() == path.parent.resolve() or output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError('Registration requires a new/empty output directory; original results are preserved')
    source_config = path.parent / 'config.json'
    config = json.loads(source_config.read_text())
    if config.get('token_attention_batch_first') is not True:
        raise RuntimeError('Legacy registration requires recorded batch_first=True; an old attention layout cannot be relabeled')
    if config.get('artifact_contract'):
        raise ValueError('This checkpoint is already versioned; use the normal checked loader')
    with dataset_lock(Path(data_root) / config['dataset']):
        features = FeatureStore(Path(data_root) / config['dataset'])
        if config.get('preprocessing') != features.preprocessing:
            raise RuntimeError('Original preprocessing configuration differs from the current data')
        original_metadata = config.get('feature_metadata', {})
        for key in ('tag_ids', 'protein_dim', 'drug_dim', 'protein_truncation', 'precision'):
            if key not in original_metadata or original_metadata[key] != features.metadata.get(key):
                raise RuntimeError('Original feature metadata differs or lacks ' + key)
        config = dict(config, data_root=str(data_root), run_id=uuid.uuid4().hex)
        contract = make_contract(config, features)
        model = ColdDTIAblation(config['variant'], **contract['options']).to(device).eval()
        state = torch.load(path, map_location=device, weights_only=True)
        model.load_state_dict({key: value for key, value in state.items()
            if not key.startswith(('drug_extractor.', 'protein_extractor.'))}, strict=True)
        features.expected_splits = {name: digest for name, digest in contract['data'].items() if '/' in name}
        dataset = InteractionDataset(features.directory / config['split'] / 'test.csv', features)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, collate_fn=keep_variable_lengths)
    _, predictions = evaluate(model, loader, device)
    reference = pd.read_csv(reference_predictions, dtype={'protein_id': str})
    keys = ['smiles_nid', 'protein_nid', 'protein_id', 'Y']
    score = 'score' if 'score' in reference else 'y_score'
    if not predictions[keys].reset_index(drop=True).equals(reference[keys].reset_index(drop=True)):
        raise RuntimeError('Original prediction rows/order/labels differ from the current test snapshot')
    if not np.allclose(predictions.score, reference[score], atol=1e-6, rtol=0):
        raise RuntimeError('Legacy predictions do not replay with the current implementation and features')
    config['artifact_contract'] = contract
    config['legacy_registration'] = dict(source_checkpoint_sha256=sha256(path),
        source_config_sha256=sha256(source_config), reference_predictions_sha256=sha256(reference_predictions),
        max_probability_difference=float(np.max(np.abs(predictions.score.to_numpy() - reference[score].to_numpy()))))
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(output_dir / 'config.json', config)
    save_checkpoint(output_dir / 'best.pt', model, contract)
    return config['legacy_registration']


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Register compatible legacy weights through original prediction replay')
    parser.add_argument('--register-legacy', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--reference-predictions', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    print(json.dumps(register_legacy(args.register_legacy, args.data_root, args.reference_predictions,
                                    args.output_dir, args.device), indent=2))
