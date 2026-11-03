"""Publish small prepared tables with rollback and crash recovery."""
import json
from pathlib import Path
import shutil

from .artifacts import PUBLICATION, atomic_json


def recover(directory):
    directory = Path(directory)
    transaction = directory / PUBLICATION
    if not transaction.exists():
        return False
    manifest = transaction / 'manifest.json'
    if not manifest.exists():
        # No source file is changed until the complete backup is recorded.
        shutil.rmtree(transaction)
        return True
    state = json.loads(manifest.read_text())
    if state['status'] == 'publishing':
        atomic_json(directory / 'feature_status.json', dict(ready=False, reason='Recovering interrupted preprocessing'))
        ordered = sorted(state['files'], key=lambda name: name == 'feature_status.json')
        for name in ordered:
            target = directory / name
            if state['files'][name]:
                backup = transaction / 'backup' / name
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_name('.' + target.name + '.restore')
                shutil.copyfile(backup, temporary)
                temporary.replace(target)
            else:
                target.unlink(missing_ok=True)
        state['status'] = 'rolled_back'
        atomic_json(manifest, state)
    # A crash while deleting backups must not trigger a second rollback.
    shutil.rmtree(transaction)
    return True


def publish(staged, directory):
    staged, directory = Path(staged), Path(directory)
    transaction = directory / PUBLICATION
    transaction.mkdir()
    files = sorted(path for path in staged.rglob('*') if path.is_file() and path.suffix in ('.csv', '.json'))
    state = dict(status='publishing', files={})
    try:
        for source in files:
            name = str(source.relative_to(staged))
            target = directory / name
            state['files'][name] = target.exists() or target.is_symlink()
            if state['files'][name]:
                backup = transaction / 'backup' / name
                backup.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(target, backup)
        atomic_json(transaction / 'manifest.json', state)
        atomic_json(directory / 'feature_status.json', dict(ready=False, reason='Publishing regenerated structure tags and indices'))
        # Every replacement is atomic. The transaction marker blocks all cache
        # readers until the complete collection has been committed or restored.
        for source in sorted(files, key=lambda path: path.name == 'feature_status.json'):
            target = directory / source.relative_to(staged)
            target.parent.mkdir(parents=True, exist_ok=True)
            source.replace(target)
        state['status'] = 'committed'
        atomic_json(transaction / 'manifest.json', state)
    except BaseException:
        recover(directory)
        raise
    shutil.rmtree(transaction)
