"""Download CIFs using explicit input tables and output paths."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', default='drugbank')
    parser.add_argument('--data-root', type=Path, default=Path(__file__).resolve().parents[1] / 'data')
    parser.add_argument('--protein-csv', type=Path, help='Table containing protein_id')
    parser.add_argument('--versions', type=int, nargs='+', default=[6, 5, 4, 3, 2, 1])
    args = parser.parse_args()
    import pandas as pd
    import requests
    from tqdm import tqdm
    directory = args.data_root / args.dataset
    source = args.protein_csv or directory / 'source_proteins.csv'
    table = pd.read_csv(source, dtype={'protein_id': str})
    observations = directory / 'observations.csv'
    if observations.exists() and not args.protein_csv:
        table = table.loc[table.protein_id.isin(pd.read_csv(observations).protein_id)]
    destination = directory / 'cif'
    destination.mkdir(parents=True, exist_ok=True)
    with requests.Session() as session:
        for identifier in tqdm(table.protein_id.drop_duplicates(), desc=args.dataset):
            path = destination / (identifier + '.cif')
            if path.exists() and path.stat().st_size:
                continue
            for version in args.versions:
                url = f'https://alphafold.ebi.ac.uk/files/AF-{identifier}-F1-model_v{version}.cif'
                try:
                    response = session.get(url, timeout=30)
                except requests.RequestException as error:
                    tqdm.write(f'{identifier}: {error}')
                    continue
                if response.status_code == 200 and response.text.lstrip().startswith('data_'):
                    path.write_bytes(response.content)
                    break
            else:
                tqdm.write(f'{identifier}: no CIF downloaded')
    print('Run filter.py to validate downloaded structures; file existence is not sufficient.')


if __name__ == '__main__':
    main()
