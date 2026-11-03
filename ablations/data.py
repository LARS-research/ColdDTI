"""Read existing split CSVs and frozen token/embedding arrays."""
from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from preprocess.artifacts import (TABLES, dataset_lock, read_feature_state, sha256,
                                  validate_branch)


class FeatureStore:
    def __init__(self, directory):
        self.directory = Path(directory)
        with dataset_lock(self.directory):
            self.metadata = read_feature_state(self.directory)
            config = self.directory / 'preprocessing.json'
            self.preprocessing = json.loads(config.read_text()) if config.exists() else {}
            drug = validate_branch(self.directory, 'drug', self.metadata['branches']['drug'])
            protein = validate_branch(self.directory, 'protein', self.metadata['branches']['protein'])
            self.arrays = [drug[0], protein[0], drug[1], protein[1]]
            self.drug_ids = set(pd.read_csv(self.directory / 'smiles.csv').smiles_nid.astype(int))
            self.protein_ids = pd.read_csv(self.directory / 'protein.csv', dtype={'protein_id': str}).protein_id.tolist()
            self.expected_splits = None
            self.assert_current()

    def assert_current(self):
        state = read_feature_state(self.directory)
        if state != self.metadata:
            raise RuntimeError('Feature metadata changed after loading; restart with a consistent snapshot')
        for branch, filename in TABLES.items():
            if sha256(self.directory / filename) != self.metadata['branches'][branch]['input_sha256']:
                raise RuntimeError('Input tables changed after feature generation; run get_emb.py')

    def read_split(self, path):
        path = Path(path)
        with dataset_lock(self.directory):
            self.assert_current()
            identity = sha256(path)
            frame = pd.read_csv(path, dtype={'protein_id': str})
            relative = str(path.resolve().relative_to(self.directory.resolve()))
            if self.expected_splits is not None and self.expected_splits.get(relative) != identity:
                raise RuntimeError('Split differs from the checkpoint training snapshot: ' + relative)
            required = ['smiles_nid', 'protein_nid', 'protein_id', 'Y']
            if frame.empty or frame[required].isna().any().any() or not frame.Y.isin([0, 1]).all():
                raise ValueError('Split requires complete identities and binary labels: ' + str(path))
            for column in ('smiles_nid', 'protein_nid'):
                values = frame[column].to_numpy()
                if not np.isfinite(values).all() or (values < 0).any() or not np.equal(values, values.astype(int)).all():
                    raise ValueError('Split indices must be nonnegative integers')
            if not frame.smiles_nid.isin(self.drug_ids).all() or not frame.protein_nid.lt(len(self.protein_ids)).all():
                raise ValueError('Split identity is not present in the feature input tables')
            if [self.protein_ids[int(index)] for index in frame.protein_nid] != frame.protein_id.tolist():
                raise ValueError('Split protein indices disagree with actual protein identities')
            self.assert_current()
            if sha256(path) != identity:
                raise RuntimeError('Split changed while it was being loaded')
            return frame, identity

    def pair(self, drug, protein):
        if drug not in self.drug_ids or not 0 <= protein < len(self.protein_ids):
            raise ValueError('Unknown feature identity: drug={}, protein={}'.format(drug, protein))
        indices = (drug, protein, drug, protein)
        values = [array[index] for array, index in zip(self.arrays, indices)]
        if any(value is None for value in values):
            raise ValueError(f"Missing cached feature: drug={drug}, protein={protein} in {self.directory}")
        tensors = []
        for i, value in enumerate(values):
            tensor = torch.as_tensor(value)
            if i < 2:
                tensor = tensor.long().reshape(1, -1)
            else:
                tensor = tensor.float()
                if tensor.ndim == 2:
                    tensor = tensor.unsqueeze(0)
            tensors.append(tensor)
        return tensors


class InteractionDataset(Dataset):
    def __init__(self, csv_path, features, sample_count=None, seed=57):
        self.frame, self.source_sha256 = features.read_split(csv_path)
        if sample_count is not None:
            self.frame = self.frame.sample(n=sample_count, random_state=seed).reset_index(drop=True)
        self.rows = self.frame[["smiles_nid", "protein_nid", "Y"]].astype(int).to_numpy()
        self.features = features

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        drug, protein, label = map(int, self.rows[index])
        return self.features.pair(drug, protein), label, protein, index


def keep_variable_lengths(batch):
    return batch


def make_loaders(directory, split, features, batch_size, seed, train_samples=None):
    loaders = []
    for part in ("train", "val", "test"):
        dataset = InteractionDataset(Path(directory) / split / f"{part}.csv", features,
                                     sample_count=train_samples if part == "train" else None, seed=57)
        loaders.append(DataLoader(dataset, batch_size=batch_size, shuffle=part == "train",
                                  collate_fn=keep_variable_lengths,
                                  generator=torch.Generator().manual_seed(seed)))
    return loaders
