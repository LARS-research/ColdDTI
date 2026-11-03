"""Small manifests, atomic metadata writes and dataset artifact validation."""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import tempfile

FEATURE_SCHEMA = 1
PUBLICATION = '.preprocess-transaction'
FILES = {'drug': ('smilestokenized.npy', 'smilesembeddings.npy'),
         'protein': ('proteinstokenized.npy', 'proteinsembeddings.npy')}
TABLES = {'drug': 'smiles.csv', 'protein': 'protein.csv'}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    descriptor, name = tempfile.mkstemp(prefix='.' + path.name + '-', dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, 'w') as handle:
            json.dump(value, handle, indent=2)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def dataset_lock(directory, exclusive=False):
    """Serialize publishers; readers hold a shared lock while loading a snapshot."""
    import fcntl
    path = Path(directory) / '.artifacts.lock'
    # A portable/read-only data bundle need not contain the local lock file.
    # Readers also compare metadata before/after loading to detect replacement.
    if not exclusive and not path.exists():
        yield
        return
    with path.open('a+b' if exclusive else 'rb') as handle:
        try:
            fcntl.flock(handle, (fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH) | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('Another process is reading or updating dataset artifacts: ' + str(directory)) from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def ensure_published(directory):
    if (Path(directory) / PUBLICATION).exists():
        raise RuntimeError('Interrupted preprocessing publication in {}. Run filter.py --recover '
                           'with the same --dataset and --data-root before continuing.'.format(directory))


def read_feature_state(directory):
    ensure_published(directory)
    path = Path(directory) / 'feature_status.json'
    if not path.is_file():
        raise RuntimeError('Feature metadata is missing in {}. Run get_emb.py; unversioned caches are not accepted.'.format(directory))
    state = json.loads(path.read_text())
    if state.get('ready') is not True:
        raise RuntimeError('Features are stale in {}. Run get_emb.py after preprocessing.'.format(directory))
    if state.get('schema_version') != FEATURE_SCHEMA:
        raise RuntimeError('Feature cache has an unsupported metadata version; regenerate it with get_emb.py')
    if any(state.get(branch + '_ready') is not True or
           state.get('branches', {}).get(branch, {}).get('ready') is not True for branch in FILES):
        raise RuntimeError('Feature branch metadata is incomplete; regenerate it with get_emb.py')
    required_tags = {'secondary_start', 'secondary_end', 'tertiary_start', 'tertiary_end'}
    tags = state.get('tag_ids', {})
    if set(tags) != required_tags or not all(type(value) is int and value >= 0 for value in tags.values()) or len(set(tags.values())) != 4:
        raise RuntimeError('Feature hierarchy token metadata is invalid; regenerate it with get_emb.py')
    if state.get('feature_id') != feature_id(state):
        raise RuntimeError('Feature metadata identity is inconsistent; regenerate the cache')
    return state


def feature_id(state):
    return digest_json(dict(schema_version=state.get('schema_version'),
                            branches=state.get('branches'), tag_ids=state.get('tag_ids')))


def table_indices(directory, branch):
    import numpy as np
    import pandas as pd
    table = pd.read_csv(Path(directory) / TABLES[branch])
    if table.empty:
        raise ValueError('Feature input table is empty: ' + TABLES[branch])
    if branch == 'protein':
        if table.protein_id.isna().any() or table.protein_id.duplicated().any() or table.protein.isna().any():
            raise ValueError('Protein input needs unique identities and complete sequences')
        return table, list(range(len(table))), len(table)
    values = table.smiles_nid.to_numpy()
    if table.smiles.isna().any() or not np.isfinite(values).all() or (values < 0).any() or not np.equal(values, values.astype(int)).all():
        raise ValueError('SMILES input needs complete sequences and nonnegative integer identities')
    indices = values.astype(int).tolist()
    if not indices or len(set(indices)) != len(indices):
        raise ValueError('SMILES identities must be nonempty and unique')
    return table, indices, max(indices) + 1


def validate_arrays(tokens, hidden, indices, size, dimension):
    import numpy as np
    if tokens.ndim != 1 or hidden.ndim != 1 or len(tokens) != size or len(hidden) != size:
        raise ValueError('Feature entity counts do not match the input table')
    for index in indices:
        token, embedding = tokens[index], hidden[index]
        if token is None or embedding is None:
            raise ValueError('Missing feature for entity index ' + str(index))
        token, embedding = np.asarray(token), np.asarray(embedding)
        if token.ndim != 2 or token.shape[0] != 1 or token.shape[1] < 1 or not np.issubdtype(token.dtype, np.integer) or (token < 0).any():
            raise ValueError('Invalid token array at entity index ' + str(index))
        if embedding.shape != (1, token.shape[1], dimension) or embedding.dtype != np.float32 or not np.isfinite(embedding).all():
            raise ValueError('Embedding shape/precision/finite-value mismatch at entity index ' + str(index))


def validate_branch(directory, branch, record, arrays=None):
    import numpy as np
    directory = Path(directory)
    if record.get('ready') is not True or not record.get('encoder'):
        raise ValueError(branch + ' features lack complete encoder provenance')
    _, indices, size = table_indices(directory, branch)
    if record.get('input_sha256') != sha256(directory / TABLES[branch]):
        raise ValueError(branch + ' features do not correspond to the current input table')
    if record.get('entities') != len(indices) or record.get('size') != size:
        raise ValueError(branch + ' feature indexing differs from the input table')
    for name in FILES[branch]:
        path, expected = directory / name, record.get('files', {}).get(name, {})
        if not path.is_file() or path.stat().st_size != expected.get('bytes') or sha256(path) != expected.get('sha256'):
            raise ValueError('Feature file is missing or changed: ' + str(path))
    if arrays is None:
        arrays = [np.load(directory / name, allow_pickle=True) for name in FILES[branch]]
    validate_arrays(*arrays, indices, size, record['dimension'])
    return arrays


def encoder_identity(directory):
    directory = Path(directory)
    names = {'config.json', 'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json',
             'added_tokens.json', 'vocab.txt', 'vocab.json', 'merges.txt', 'structure_tokens.json'}
    paths = [path for path in directory.iterdir() if path.is_file() and
             (path.name in names or path.suffix in ('.bin', '.safetensors') and path.name != 'training_args.bin'
              or path.name.endswith('.index.json'))]
    if not (directory / 'config.json').is_file() or not any(path.suffix in ('.bin', '.safetensors') for path in paths):
        raise ValueError('Encoder configuration or weights are missing: ' + str(directory))
    result = {}
    for path in sorted(paths):
        if path.name in ('config.json', 'tokenizer_config.json'):
            config = json.loads(path.read_text())
            # Hugging Face records the local load path in saved configurations;
            # it does not change either encoder or tokenizer behavior.
            for key in ('_name_or_path', 'name_or_path'):
                config.pop(key, None)
            result[path.name] = digest_json(config)
        else:
            result[path.name] = sha256(path)
    return result
