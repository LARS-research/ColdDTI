from Bio.PDB import MMCIFParser, PPBuilder
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from collections import defaultdict
import pandas as pd
from tqdm import tqdm
import argparse
import os

def load_structure(cif_file):
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("structure", cif_file)
    return structure

def extract_sequences(structure):
    """提取所有链的氨基酸序列和残基位置"""
    ppb = PPBuilder()
    seqs = {}
    res_maps = {}

    for model in structure:
        for chain in model:
            sequence = ""
            res_ids = []
            for pp in ppb.build_peptides(chain):
                sequence += str(pp.get_sequence())
                for res in pp:
                    res_ids.append(res.get_id()[1])
            if sequence:
                seqs[chain.id] = sequence
                res_maps[chain.id] = res_ids
    return seqs, res_maps

def extract_secondary_structure_blocks(cif_file):
    """提取结构段为字典列表形式：{'type': 'H', 'start': x, 'end': y}"""
    cif_dict = MMCIF2Dict(cif_file)
    conf_types = cif_dict.get('_struct_conf.conf_type_id', [])
    beg_chain = cif_dict.get('_struct_conf.beg_label_asym_id', [])
    beg_res = cif_dict.get('_struct_conf.beg_label_seq_id', [])
    end_chain = cif_dict.get('_struct_conf.end_label_asym_id', [])
    end_res = cif_dict.get('_struct_conf.end_label_seq_id', [])

    ss_blocks = defaultdict(list)

    for i in range(len(conf_types)):
        chain_id = beg_chain[i]
        if chain_id != end_chain[i]:
            continue
        try:
            start = int(beg_res[i])
            end = int(end_res[i])
        except ValueError:
            continue
        ss_type = ss_type_to_char(conf_types[i])
        ss_blocks[chain_id].append({
            'type': ss_type,
            'start': start,
            'end': end
        })

    return ss_blocks

def ss_type_to_char(ss_type):
    if "HELX" in ss_type:
        return "H"
    elif "SHEET" in ss_type:
        return "E"
    elif "TURN" in ss_type:
        return "T"
    elif "BEND" in ss_type:
        return "B"
    else:
        return "C"

def extract_hierarchical(cif_file):
    structure = load_structure(cif_file)
    sequences, _ = extract_sequences(structure)
    ss_block_data = extract_secondary_structure_blocks(cif_file)
    hierarchical_structure = {}
    for k in sequences.keys():
        hierarchical_structure[k] = {
            "primary": sequences[k],
            "secondary": ss_block_data[k]
        }
    return hierarchical_structure

def add_space_between_uppercase(s):
    result = []
    i = 0
    while i < len(s):
        if s[i].isupper():
            j = i
            while j < len(s) and s[j].isupper():
                j += 1
            length = j - i
            if length > 1:
                result.append(' '.join(s[i:j]))
            else:
                result.append(s[i])
            i = j
        else:
            result.append(s[i])
            i += 1
    return ''.join(result)

def process(structure):
    complete_seq = ""
    for _, v in structure.items():
        seq = v['primary']
        insert_len = 0
        for sec in v['secondary']:
            start = sec['start'] + insert_len - 1
            seq = seq[: start] + '[secondary_start]' + f'[{sec["type"]}]' + seq[start:]
            insert_len += 20

            end = sec['end'] + insert_len
            seq = seq[: end] + '[secondary_end]' + seq[end:]
            insert_len += 15

        complete_seq += "[tertiary_start]" + seq + "[tertiary_end]"
        complete_seq = add_space_between_uppercase(complete_seq)
    return complete_seq

def fallback_marked_from_plain(plain_seq: str):
    """
    ### >>> CHANGE
    没有 CIF / 解析失败时的 fallback：
    仍然生成一个可被 tokenizer 处理的 marked 序列，保证 protein.csv 不缺行、不漂移。
    ### <<< CHANGE
    """
    if plain_seq is None:
        plain_seq = ""
    plain_seq = str(plain_seq).strip().replace(" ", "")
    marked = "[tertiary_start]" + plain_seq + "[tertiary_end]"
    return add_space_between_uppercase(marked)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='settings')
    parser.add_argument('--dataset', type=str, default="drugbank",
                        choices=["drugbank", "kiba", "davis", "bindingdb", "biosnap", "human"],
                        help='select dataset for processing')
    parser.add_argument('--types', nargs="+", default=["real_timeline"],
                        help="which split types to rebuild, e.g. --types real_timeline cold_pair")
    args = parser.parse_args()

    # =========================
    # 1) Load base protein table
    # =========================
    if args.dataset in ['drugbank', 'kiba', 'bindingdb', 'biosnap', 'human']:
        df_proteins = pd.read_csv(f'../Dataset/{args.dataset}/protein_with_timestamps.csv')
        # 需要列：protein_id, protein(plain seq)
        assert "protein_id" in df_proteins.columns and "protein" in df_proteins.columns
    elif args.dataset == 'davis':
        df_proteins = pd.read_csv(f'./data/davis/protein_uniprot.csv')
        assert "protein_id" in df_proteins.columns and "protein" in df_proteins.columns

    # =========================
    # 2) Build ./data/{dataset}/protein.csv (marked)
    # =========================
    pids = []
    seqs = []

    n_total = len(df_proteins)
    n_cif_ok = 0
    n_fallback = 0

    for i in tqdm(range(n_total), desc=f"[build protein.csv] {args.dataset}"):
        pid = str(df_proteins.iloc[i]['protein_id']).strip()
        plain_seq = df_proteins.iloc[i]['protein']

        cif_path = f"./data/{args.dataset}/cif/{pid}.cif"
        use_fallback = False

        # ### >>> CHANGE
        # 以前：没 cif 就 continue（导致 protein.csv 缺行、nid 漂移）
        # 现在：没 cif / 空文件 -> fallback（仍然写入 protein.csv）
        # ### <<< CHANGE
        if (not os.path.exists(cif_path)) or os.path.getsize(cif_path) == 0:
            use_fallback = True
        else:
            try:
                hierarchical_structure = extract_hierarchical(cif_path)
                process_seq = process(hierarchical_structure)
            except Exception:
                use_fallback = True

        if use_fallback:
            process_seq = fallback_marked_from_plain(plain_seq)
            n_fallback += 1
        else:
            n_cif_ok += 1

        pids.append(pid)
        seqs.append(process_seq)

    new_df_proteins = pd.DataFrame({
        'protein_id': pids,
        'protein': seqs
    })
    out_protein_csv = f'./data/{args.dataset}/protein.csv'
    new_df_proteins.to_csv(out_protein_csv, index=False)
    print(f"[OK] wrote {out_protein_csv} (N={len(new_df_proteins)}, cif_ok={n_cif_ok}, fallback={n_fallback})")

    # new nid mapping: row index in ./data/{dataset}/protein.csv
    pid2nid = {pid: nid for nid, pid in enumerate(new_df_proteins['protein_id'].tolist())}

    # davis mapping (保留你原逻辑)
    if args.dataset == 'davis':
        map_csv = './data/davis/mapping_davis_strict.csv'
        df_map = pd.read_csv(map_csv)
        sym2uni = dict(zip(df_map['protein_symbol'], df_map['uniprot_id']))
    else:
        sym2uni = None

    # =========================
    # 3) Rebuild split csv with CORRECT protein_nid
    # =========================
    for type_name in args.types:
        for split in ['train', 'val', 'test']:
            out_dir = f'./data/{args.dataset}/{type_name}'
            os.makedirs(out_dir, exist_ok=True)

            in_csv = f'../Dataset/{args.dataset}/{type_name}/{split}.csv'
            if not os.path.exists(in_csv):
                print(f"[WARN] missing {in_csv}, skip.")
                continue

            df_in = pd.read_csv(in_csv)

            # 你 Dataset 的列可能是：
            # smiles_nid, protein_nid, smiles_id, protein_id, smiles, protein, Y
            # 我们只强依赖：smiles_nid, protein_id, Y
            assert "smiles_nid" in df_in.columns and "protein_id" in df_in.columns and "Y" in df_in.columns

            s_nids = []
            p_nids = []
            p_ids = []
            Ys = []
            dropped = 0

            for i in range(len(df_in)):
                s_nid = int(df_in.iloc[i]['smiles_nid'])
                p_id = str(df_in.iloc[i]['protein_id']).strip()
                y = int(df_in.iloc[i]['Y'])

                if args.dataset == 'davis':
                    # symbol -> UniProt
                    p_id_uni = sym2uni.get(p_id)
                    if p_id_uni is None:
                        dropped += 1
                        continue
                    p_id = p_id_uni

                # ### >>> CHANGE
                # 关键：protein_nid 必须来自 pid2nid（即 ./data/{dataset}/protein.csv 的行号）
                # 不再使用 Dataset 里的旧 protein_nid
                # ### <<< CHANGE
                p_nid_new = pid2nid.get(p_id, None)
                if p_nid_new is None:
                    dropped += 1
                    continue

                # 可选：强制一致性断言（防止未来再漂）
                if new_df_proteins.iloc[p_nid_new]["protein_id"] != p_id:
                    raise RuntimeError(
                        f"[ALIGN ERROR] pid2nid mismatch: p_id={p_id}, "
                        f"p_nid_new={p_nid_new}, "
                        f"protein.csv@nid={new_df_proteins.iloc[p_nid_new]['protein_id']}"
                    )

                s_nids.append(s_nid)
                p_nids.append(int(p_nid_new))
                p_ids.append(p_id)
                Ys.append(y)

            df_out = pd.DataFrame({
                'smiles_nid': s_nids,
                'protein_nid': p_nids,
                'protein_id': p_ids,
                'Y': Ys
            })

            out_csv = f'{out_dir}/{split}.csv'
            df_out.to_csv(out_csv, index=False)
            print(f"[OK] wrote {out_csv} (N={len(df_out)}, dropped={dropped})")

    print("\n[DONE] Next steps:")
    print("1) Re-run get_emb.py to regenerate proteinstokenized.npy + proteinsembeddings.npy (MUST).")
    print("2) Then re-train main_earlystop.py with the same split name (e.g., real_timeline).")