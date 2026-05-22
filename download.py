#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import requests
import os
from tqdm import tqdm
import pandas as pd

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='settings')
    parser.add_argument(
        '--dataset',
        type=str,
        default="drugbank",
        choices=['drugbank', 'davis', 'kiba', 'bindingdb', 'biosnap', 'human'],
        help='dataset'
    )
    args = parser.parse_args()

    header = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
                      'AppleWebKit/537.36 (KHTML, like Gecko) '
                      'Chrome/114.0.0.0 Safari/537.36',
    }

    # ----------------- 准备目录 & 读取 protein.csv -----------------
    if args.dataset in ['drugbank', 'kiba', 'davis', 'bindingdb', 'biosnap', 'human']:
        os.makedirs('./data', exist_ok=True)
        os.makedirs(f'./data/{args.dataset}', exist_ok=True)
        os.makedirs(f'./data/{args.dataset}/cif', exist_ok=True)

        if args.dataset in ['drugbank', 'kiba', 'bindingdb', 'biosnap', 'human']:
            df_protein = pd.read_csv(f'../Dataset/{args.dataset}/protein.csv')
        elif args.dataset == 'davis':
            df_protein = pd.read_csv(f'./data/davis/protein_uniprot.csv')
    else:
        raise NotImplementedError

    out_dir = f'./data/{args.dataset}/cif'
    print(".cif file downloading from AlphaFold (try v6→v1)...")

    # 为了兼容不同版本，依次尝试这些版本
    VERSION_LIST = [6, 5, 4, 3, 2, 1]

    # ----------------- 主循环 -----------------
    for i in tqdm(range(len(df_protein))):
        pid = str(df_protein.iloc[i]['protein_id']).strip()

        out_path = os.path.join(out_dir, f"{pid}.cif")
        # 已下过就跳过
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            continue

        success = False
        for v in VERSION_LIST:
            url = f"https://alphafold.ebi.ac.uk/files/AF-{pid}-F1-model_v{v}.cif"
            try:
                resp = requests.get(url=url, headers=header, timeout=30)
            except Exception as e:
                print(f"\n[REQUEST ERROR] {pid} v{v}: {e}")
                continue

            if resp.status_code != 200:
                # 404 / 403 之类，换下一版
                continue

            text_head = resp.text[:200]
            if "NoSuchKey" in text_head or "<Error>" in text_head:
                # S3 返回的 XML 错误页，不是 CIF
                continue

            # 到这里说明真的拿到 CIF 了
            with open(out_path, 'wb') as f:
                f.write(resp.content)
            success = True
            # 可以打印一下用的是哪个版本（可选）
            # print(f"\n[OK] {pid} -> v{v}")
            break

        if not success:
            # 6→1 都没成功，说明确实没找到 CIF
            print(f"\n[No CIF] {pid}")