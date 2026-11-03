"""Generate native NumPy frozen encoder features for one prepared dataset."""
import argparse
import json
from pathlib import Path
import tempfile

from .artifacts import (FEATURE_SCHEMA, FILES, TABLES, atomic_json, dataset_lock,
                        encoder_identity, ensure_published, feature_id, sha256,
                        table_indices, validate_arrays, validate_branch)


def encode_dataset(directory, model_root, device='cpu', seed=57, kind='all'):
    directory = Path(directory)
    with dataset_lock(directory, exclusive=True):
        ensure_published(directory)
        return _encode_dataset(directory, Path(model_root), device, seed, kind)


def _encode_dataset(directory, model_root, device, seed, kind):
    import numpy as np
    import torch
    from transformers import AutoModel, AutoTokenizer

    torch.set_num_threads(4)
    torch.manual_seed(seed)
    if device.startswith('cuda'):
        torch.cuda.manual_seed_all(seed)
    state_file = directory / 'feature_status.json'
    state = json.loads(state_file.read_text()) if state_file.exists() else {}
    if state.get('schema_version') != FEATURE_SCHEMA:
        state = {}
    requested = ('protein', 'drug') if kind == 'all' else (kind,)
    state['schema_version'] = FEATURE_SCHEMA
    state['ready'] = False
    state.pop('feature_id', None)
    for branch in requested:
        state[branch + '_ready'] = False
        state.setdefault('branches', {})[branch] = dict(ready=False)
    atomic_json(state_file, state)
    for branch in requested:
        is_protein = branch == 'protein'
        checkpoint = model_root / ('prot_resize' if is_protein else 'ChemBERTa-77M-MLM')
        provenance = encoder_identity(checkpoint)
        input_sha = sha256(directory / TABLES[branch])
        tokenizer = AutoTokenizer.from_pretrained(str(checkpoint), local_files_only=True)
        model = AutoModel.from_pretrained(str(checkpoint), local_files_only=True).eval().to(device)
        model.requires_grad_(False)
        table, indices, size = table_indices(directory, branch)
        if is_protein:
            texts = table.protein.tolist()
            tag_ids = {name: tokenizer.convert_tokens_to_ids('[' + name + ']')
                       for name in ('secondary_start', 'secondary_end', 'tertiary_start', 'tertiary_end')}
            if len(set(tag_ids.values())) != 4 or tokenizer.unk_token_id in tag_ids.values():
                raise ValueError('Protein tokenizer lacks the four registered hierarchy tokens; run resize.py')
            state['tag_ids'] = tag_ids
        else:
            texts = table.smiles.tolist()
        tokens, hidden = np.full(size, None, dtype=object), np.full(size, None, dtype=object)
        with torch.inference_mode():
            for number, (index, text) in enumerate(zip(indices, texts), 1):
                options = {} if is_protein else dict(max_length=512, truncation=True)
                inputs = tokenizer(text, return_tensors='pt', **options)
                output = model(**{key: value.to(device) for key, value in inputs.items()}, return_dict=True)
                tokens[index] = inputs.input_ids.numpy().copy()
                hidden[index] = output.last_hidden_state.float().cpu().numpy().copy()
                if number % 100 == 0 or number == len(indices):
                    print(f'{directory.name} {branch}: {number}/{len(indices)}', flush=True)
        validate_arrays(tokens, hidden, indices, size, int(model.config.hidden_size))
        if sha256(directory / TABLES[branch]) != input_sha:
            raise RuntimeError('Input table changed during feature generation; rerun this branch')
        with tempfile.TemporaryDirectory(prefix='.feature-', dir=directory) as temporary:
            staged = Path(temporary)
            file_records = {}
            for filename, array in zip(FILES[branch], (tokens, hidden)):
                path = staged / filename
                np.save(path, array, allow_pickle=True)
                file_records[filename] = dict(bytes=path.stat().st_size, sha256=sha256(path))
            # Both files are complete before either is installed. The branch
            # remains invalid until the pair and its manifest are committed.
            for filename in FILES[branch]:
                (staged / filename).replace(directory / filename)
        state['branches'][branch] = dict(ready=True, input_sha256=input_sha, encoder=provenance,
            files=file_records, dimension=int(model.config.hidden_size), entities=len(indices),
            size=size, seed=seed, precision='float32', truncation=False if is_protein else 512)
        state.update({branch + '_ready': True, branch + '_checkpoint': str(checkpoint),
                      branch + '_dim': int(model.config.hidden_size), branch + '_entities': len(indices)})
        atomic_json(state_file, state)
        del model, tokens, hidden
        if device.startswith('cuda'):
            torch.cuda.empty_cache()
    for branch in ('protein', 'drug'):
        if branch not in requested and state.get(branch + '_ready'):
            try:
                validate_branch(directory, branch, state['branches'][branch])
            except (OSError, ValueError, KeyError) as error:
                state[branch + '_ready'] = False
                state['branches'][branch]['ready'] = False
                state['reason'] = str(error)
    state.update(ready=bool(state.get('protein_ready') and state.get('drug_ready')),
                 seed=seed, precision='float32', protein_truncation=False)
    if state['ready']:
        state.pop('reason', None)
        state['feature_id'] = feature_id(state)
    else:
        state.setdefault('reason', 'Both drug and protein branches must be generated with this metadata version')
    atomic_json(state_file, state)
    return state


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', nargs='+', default=['drugbank'])
    parser.add_argument('--data-root', type=Path, default=Path(__file__).resolve().parents[1] / 'data')
    parser.add_argument('--model-root', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--seed', type=int, default=57)
    parser.add_argument('--kind', choices=('all', 'protein', 'drug'), default='all')
    args = parser.parse_args(argv)
    for dataset in args.dataset:
        encode_dataset(args.data_root / dataset, args.model_root, args.device, args.seed, args.kind)


if __name__ == '__main__':
    main()
