from transformers import AutoTokenizer, AutoModel
import numpy as np
from tqdm import tqdm
import pandas as pd
import argparse
import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='settings')
    parser.add_argument('--dataset', type=str, default="drugbank", choices=["drugbank", "kiba", "davis", "bindingdb", "biosnap", "human"],help='select dataset for processing')
    
    args = parser.parse_args()

    # df_proteins = pd.read_csv(f'../Dataset/{args.dataset}/protein.csv')
    if args.dataset == "drugbank" or args.dataset == "kiba" or args.dataset == 'bindingdb' or args.dataset == 'biosnap' or args.dataset == 'human':
        # df_proteins = pd.read_csv(f'../Dataset/{args.dataset}/protein.csv')
        df_proteins = pd.read_csv(f'./data/{args.dataset}/protein.csv')
    elif args.dataset == "davis":
        df_proteins = pd.read_csv(f'./data/davis/protein_uniprot.csv')


    protein_len = len(df_proteins)
    pnid2pid = df_proteins['protein_id'].to_dict()
    pid2pnid = {v:k for k, v in pnid2pid.items()}


    protein_content = np.empty(protein_len, dtype=object)
    protein_tokenized = np.empty(protein_len, dtype=object)
    tokenizer = AutoTokenizer.from_pretrained('../Model/prot_resize')
    model = AutoModel.from_pretrained('../Model/prot_resize').cuda()

    df_proteins = pd.read_csv(f'./data/{args.dataset}/protein.csv')
    # df_proteins = pd.read_csv(f'../Dataset/{args.dataset}/protein_with_timestamps.csv')

    # 关闭梯度跟踪，显存会小很多；不影响输出值
    with torch.inference_mode():
        for i in tqdm(range(len(df_proteins))):
            pid = df_proteins.iloc[i]['protein_id']
            seq = df_proteins.iloc[i]['protein']
            input_ids = tokenizer(seq, return_tensors='pt').input_ids.cuda()
            output = model(input_ids, return_dict=True, output_hidden_states=True).hidden_states[-1]
            
            protein_tokenized[pid2pnid[pid]] = input_ids.cpu()
            protein_content[pid2pnid[pid]] = output.cpu()
            
            # 及时释放 GPU 张量，降低峰值显存
            del input_ids, output
            torch.cuda.empty_cache()

    protein_content = np.array(protein_content, dtype=object)
    np.save(f'./data/{args.dataset}/proteinstokenized.npy', protein_tokenized)
    np.save(f'./data/{args.dataset}/proteinsembeddings.npy', protein_content)


    df_smiles = pd.read_csv(f'../Dataset/{args.dataset}/smiles.csv')
    smiles_len = len(df_smiles)
    smiles_content = np.empty(smiles_len + 1, dtype=object)
    smiles_tokenized = np.empty(smiles_len + 1, dtype=object)
    tokenizer = AutoTokenizer.from_pretrained('../Model/ChemBERTa-77M-MLM')
    model = AutoModel.from_pretrained('../Model/ChemBERTa-77M-MLM').cuda()

    with torch.inference_mode():
        for i in tqdm(range(len(df_smiles))):
            s_nid = df_smiles.iloc[i]['smiles_nid']
            seq = df_smiles.iloc[i]['smiles']
            input_ids = tokenizer(seq, return_tensors='pt', max_length=512, truncation=True).input_ids.cuda()
            output = model(input_ids, return_dict=True, output_hidden_states=True).hidden_states[-1]
            
            smiles_tokenized[s_nid] = input_ids.cpu()
            smiles_content[s_nid] = output.cpu()

    smiles_content = np.array(smiles_content, dtype=object)
    np.save(f'./data/{args.dataset}/smilestokenized.npy', smiles_tokenized)
    np.save(f'./data/{args.dataset}/smilesembeddings.npy', smiles_content)