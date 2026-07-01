from Bio.PDB import MMCIFParser, PPBuilder
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from collections import defaultdict
import pandas as pd
from tqdm import tqdm
import argparse
import os

def load_structure(cif_file):
    # Structural annotations below use label_asym_id/label_seq_id, so parse
    # chains and residues with the same mmCIF identifier convention.
    parser = MMCIFParser(
        QUIET=True,
        auth_chains=False,
        auth_residues=False,
    )
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

def _as_list(value):
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _append_structure_blocks(
    ss_blocks,
    seen,
    block_types,
    beg_chains,
    beg_residues,
    end_chains,
    end_residues,
    forced_type=None,
):
    columns = [
        _as_list(block_types),
        _as_list(beg_chains),
        _as_list(beg_residues),
        _as_list(end_chains),
        _as_list(end_residues),
    ]
    if not any(columns):
        return

    lengths = {len(column) for column in columns}
    if len(lengths) != 1:
        raise ValueError("Inconsistent secondary-structure annotation columns")

    for raw_type, beg_chain, beg_res, end_chain, end_res in zip(*columns):
        if beg_chain != end_chain:
            continue
        try:
            start = int(beg_res)
            end = int(end_res)
        except (TypeError, ValueError):
            continue
        if start < 1 or end < start:
            continue

        ss_type = forced_type or ss_type_to_char(str(raw_type))
        key = (beg_chain, start, end, ss_type)
        if key in seen:
            continue
        seen.add(key)
        ss_blocks[beg_chain].append({
            'type': ss_type,
            'start': start,
            'end': end,
        })


def extract_secondary_structure_blocks(cif_file):
    """Extract helix/turn/bend and beta-sheet residue ranges from mmCIF."""
    cif_dict = MMCIF2Dict(cif_file)
    ss_blocks = defaultdict(list)
    seen = set()

    _append_structure_blocks(
        ss_blocks,
        seen,
        cif_dict.get('_struct_conf.conf_type_id'),
        cif_dict.get('_struct_conf.beg_label_asym_id'),
        cif_dict.get('_struct_conf.beg_label_seq_id'),
        cif_dict.get('_struct_conf.end_label_asym_id'),
        cif_dict.get('_struct_conf.end_label_seq_id'),
    )

    # Beta-sheet ranges are stored in their own mmCIF category.
    sheet_beg_chains = cif_dict.get('_struct_sheet_range.beg_label_asym_id')
    _append_structure_blocks(
        ss_blocks,
        seen,
        ['SHEET'] * len(_as_list(sheet_beg_chains)),
        sheet_beg_chains,
        cif_dict.get('_struct_sheet_range.beg_label_seq_id'),
        cif_dict.get('_struct_sheet_range.end_label_asym_id'),
        cif_dict.get('_struct_sheet_range.end_label_seq_id'),
        forced_type='E',
    )

    for blocks in ss_blocks.values():
        blocks.sort(key=lambda block: (block['start'], block['end'], block['type']))
    return ss_blocks

def ss_type_to_char(ss_type):
    ss_type = ss_type.upper()
    if "HELX" in ss_type:
        return "H"
    elif "SHEET" in ss_type or "STRN" in ss_type:
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
    if not sequences:
        raise ValueError("No peptide sequence recovered from structure")
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
    n_missing_cif = 0
    n_empty_cif = 0
    n_parse_failed = 0

    for i in tqdm(range(n_total), desc=f"[build protein.csv] {args.dataset}"):
        pid = str(df_proteins.iloc[i]['protein_id']).strip()
        cif_path = f"./data/{args.dataset}/cif/{pid}.cif"

        if not os.path.exists(cif_path):
            n_missing_cif += 1
            continue
        if os.path.getsize(cif_path) == 0:
            n_empty_cif += 1
            continue

        try:
            hierarchical_structure = extract_hierarchical(cif_path)
            process_seq = process(hierarchical_structure)
            if not process_seq.strip():
                raise ValueError("Tagged protein sequence is empty")
        except Exception as exc:
            n_parse_failed += 1
            tqdm.write(f"[WARN] failed to parse {pid}: {exc}")
            continue

        pids.append(pid)
        seqs.append(process_seq)
        n_cif_ok += 1

    new_df_proteins = pd.DataFrame({
        'protein_id': pids,
        'protein': seqs
    })
    out_protein_csv = f'./data/{args.dataset}/protein.csv'
    new_df_proteins.to_csv(out_protein_csv, index=False)
    print(
        f"[OK] wrote {out_protein_csv} "
        f"(retained={n_cif_ok}, missing_cif={n_missing_cif}, "
        f"empty_cif={n_empty_cif}, parse_failed={n_parse_failed})"
    )

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
