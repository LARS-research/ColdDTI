"""Download pinned base encoders, or verify the recorded local encoder files."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def matches(path, expected):
    return (path.is_file() and path.stat().st_size == expected['bytes']
            and sha256(path) == expected['sha256'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-root', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, default=ROOT / 'resources/encoders.json')
    parser.add_argument('--models', nargs='+', default=['prot_bert', 'ChemBERTa-77M-MLM'])
    parser.add_argument('--check', action='store_true', help='Verify files without network access or writes')
    args = parser.parse_args(argv)
    manifest = json.loads(args.manifest.read_text())
    failures = []
    for name in args.models:
        if name not in manifest:
            parser.error('Unknown encoder: ' + name)
        entry = manifest[name]
        if not args.check and not entry.get('repo_id'):
            parser.error(name + ' is a local expanded encoder; use resize.py to create a new one')
        for filename, expected in entry['files'].items():
            target = args.model_root / name / filename
            if matches(target, expected):
                print('OK', name + '/' + filename, flush=True)
                continue
            if args.check:
                failures.append(str(target))
                continue
            if target.exists() or target.is_symlink():
                raise ValueError('Existing file differs from the recorded encoder; choose a new model root: ' + str(target))
            import requests
            url = 'https://huggingface.co/{}/resolve/{}/{}'.format(
                entry['repo_id'], entry['revision'], filename)
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name('.' + filename + '.partial')
            with requests.get(url, stream=True, timeout=(30, 120)) as response:
                response.raise_for_status()
                with temporary.open('wb') as handle:
                    for block in response.iter_content(1024 * 1024):
                        handle.write(block)
            if not matches(temporary, expected):
                raise ValueError('Downloaded encoder verification failed: ' + str(temporary))
            temporary.replace(target)
            print('DOWNLOADED', name + '/' + filename, flush=True)
    if failures:
        raise SystemExit('Missing or different encoder files:\n' + '\n'.join(failures))


if __name__ == '__main__':
    main()
