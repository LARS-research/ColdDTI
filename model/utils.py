from random import shuffle, choice
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader

class Mydataset(Dataset):
    def __init__(self, samples, smiles_tokenized, proteins_tokenized, smiles_content, proteins_content):


        self.samples = samples
        self.len = len(samples)
        self.smiles_tokenized = smiles_tokenized
        self.proteins_tokenized = proteins_tokenized
        self.smiles_content = smiles_content
        self.proteins_content = proteins_content
        # 预过滤：只保留 4 个对象都不是 None 的样本
        valid_samples = []
        dropped = 0

        for row in samples:
            s_nid = int(row[0])
            p_nid = int(row[1])

            # 索引越界也踢掉
            if s_nid >= len(smiles_tokenized) or p_nid >= len(proteins_tokenized):
                dropped += 1
                continue

            s_tok = smiles_tokenized[s_nid]
            p_tok = proteins_tokenized[p_nid]
            s_emb = smiles_content[s_nid]
            p_emb = proteins_content[p_nid]

            if s_tok is None or p_tok is None or s_emb is None or p_emb is None:
                dropped += 1
                continue

            valid_samples.append(row)

        self.samples = np.array(valid_samples)
        self.len = len(self.samples)

        print(f"[Mydataset] kept {self.len} samples, dropped {dropped} invalid samples.")

    def __getitem__(self, idx):
        one_sample = self.samples[idx]
        return self.smiles_tokenized[int(one_sample[0])], self.proteins_tokenized[int(one_sample[1])], self.smiles_content[int(one_sample[0])], self.proteins_content[int(one_sample[1])], torch.tensor([self.samples[idx,-1]]), one_sample
    def __len__(self):
        return self.len

def load_tensor(file_name, dtype):
    print(f"Loading {file_name}...")
    return [dtype(d) if d is not None else d for d in np.load(file_name + '.npy', allow_pickle=True)]

def preparedata(type, dataset):
    dir_input = ('./data/' + dataset + '/')
    smiles_content = load_tensor(dir_input + 'smilesembeddings', torch.FloatTensor)
    proteins_content = load_tensor(dir_input + 'proteinsembeddings', torch.FloatTensor) 
    smiles_tokenized = load_tensor(dir_input + 'smilestokenized', torch.LongTensor)
    proteins_tokenized = load_tensor(dir_input + 'proteinstokenized', torch.LongTensor)
    trainfiles = pd.read_csv(dir_input + type + '/train.csv')
    validfiles = pd.read_csv(dir_input + type + '/val.csv')
    testfiles = pd.read_csv(dir_input + type + '/test.csv')
    

    return trainfiles.values, validfiles.values, testfiles.values, smiles_tokenized, proteins_tokenized, smiles_content, proteins_content


def collatef(batch):
    batchlist = []
    for item in batch:
        smiles_tokenized, proteins_tokenized, smiles_content, proteins_content, interaction, sample = item
        list=[smiles_tokenized, proteins_tokenized, smiles_content, proteins_content, interaction, sample]
        batchlist.append(list)
    return batchlist

def preparedataset(batch_size,type,dataset):
    trainsamples, validsamples, testsamples, smiles_tokenized, proteins_tokenized, smiles_content, proteins_content = preparedata(type, dataset)
    trainloader = DataLoader(Mydataset(trainsamples, smiles_tokenized, proteins_tokenized, smiles_content, proteins_content), shuffle = True, batch_size = batch_size, collate_fn = collatef, drop_last = False)
    validloader = DataLoader(Mydataset(validsamples, smiles_tokenized, proteins_tokenized, smiles_content, proteins_content), shuffle = False, batch_size = batch_size,
                            collate_fn = collatef, drop_last = False)
    testloader = DataLoader(Mydataset(testsamples, smiles_tokenized, proteins_tokenized, smiles_content, proteins_content), shuffle = False, batch_size = batch_size,
                             collate_fn = collatef, drop_last = False)
    return trainloader, validloader, testloader, smiles_tokenized, proteins_tokenized, smiles_content, proteins_content