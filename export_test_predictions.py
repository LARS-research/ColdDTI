# -*- coding: utf-8 -*-
import os
import argparse
import random
import numpy as np
import pandas as pd
import torch
from transformers import AutoModel

from model.model_pro import Config
from model.utils import preparedataset
from model.colddti import ColdDTI


def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_test_columns(dataset: str, split: str):
    """
    Read ./data/{dataset}/{split}/test.csv header columns.
    If missing, return None (fallback to col_0/col_1/...)
    """
    test_csv = f"./data/{dataset}/{split}/test.csv"
    if not os.path.exists(test_csv):
        print(f"[WARN] cannot find {test_csv}, will use col_0/col_1/... as sample columns.")
        return None
    df = pd.read_csv(test_csv)
    return list(df.columns)


def _safe_int(x):
    try:
        return int(x)
    except Exception:
        return None


@torch.no_grad()
def export_predictions(
    model: ColdDTI,
    testloader,
    device,
    out_csv: str,
    out_pos_csv: str,
    sample_columns=None,
    ablation=None,
    protein_df_marked=None,       # ./data/{dataset}/protein.csv  (model-used protein order)
    drugbank_map_df=None,         # NEW: smiles_with_drugid.csv (smiles_nid -> drugbank_id)
):
    model.eval()
    rows = []

    # testloader yields: item(list) length=batch_size
    # each data: smiles_tokenized, proteins_tokenized, smiles_content, proteins_content, interaction, sample_row
    for item in testloader:
        for data in item:
            smiles_tokenized, proteins_tokenized, smiles_content, proteins_content, interaction, sample_row = data

            smiles_tokenized = smiles_tokenized.to(device)
            proteins_tokenized = proteins_tokenized.to(device)
            smiles_content = smiles_content.to(device)
            proteins_content = proteins_content.to(device)
            interaction = interaction.to(device)

            correct_label, predicted_label, predicted_score = model(
                smiles_tokenized, proteins_tokenized,
                smiles_content, proteins_content,
                interaction,
                train=False,
                ablation=ablation
            )

            # sample_row from csv row
            if isinstance(sample_row, np.ndarray):
                sample_vals = sample_row.tolist()
            else:
                sample_vals = list(sample_row)

            # align column names
            if sample_columns is None:
                sample_columns_now = [f"col_{i}" for i in range(len(sample_vals))]
            else:
                if len(sample_columns) != len(sample_vals):
                    sample_columns_now = sample_columns[:len(sample_vals)]
                    if len(sample_columns_now) < len(sample_vals):
                        sample_columns_now += [f"col_{i}" for i in range(len(sample_columns_now), len(sample_vals))]
                else:
                    sample_columns_now = sample_columns

            row_dict = {sample_columns_now[i]: sample_vals[i] for i in range(len(sample_vals))}
            row_dict.update({
                "y_true": int(correct_label),
                "y_pred": int(predicted_label),
                "y_score": float(predicted_score),
            })

            # =========================
            # NEW(1): parse nids robustly
            # =========================
            s_nid = _safe_int(row_dict.get("smiles_nid", row_dict.get("col_0", None)))
            p_nid = _safe_int(row_dict.get("protein_nid", row_dict.get("col_1", None)))

            # =========================
            # NEW(2): protein_id actually used by model
            # =========================
            if (protein_df_marked is not None) and (p_nid is not None) and (0 <= p_nid < len(protein_df_marked)):
                row_dict["protein_id_used_by_model"] = protein_df_marked.iloc[p_nid]["protein_id_used_by_model"]
                row_dict["protein_marked_prefix_used_by_model"] = str(
                    protein_df_marked.iloc[p_nid]["protein_marked"]
                )[:120]
            else:
                row_dict["protein_id_used_by_model"] = None
                row_dict["protein_marked_prefix_used_by_model"] = None

            # =========================
            # NEW(3): drugbank_id used by model (from smiles_with_drugid.csv)
            # =========================
            # 你要的是 DBxxxx，所以用 smiles_nid 去 drugbank_map_df 查
            if (drugbank_map_df is not None) and (s_nid is not None):
                hit = drugbank_map_df.loc[drugbank_map_df["smiles_nid"] == s_nid]
                if len(hit) > 0:
                    row_dict["drugbank_id_used_by_model"] = hit["drugbank_id"].iloc[0]
                    # 可选：也把 smiles 对上（方便 sanity check）
                    if "smiles" in hit.columns:
                        row_dict["smiles_used_by_model"] = hit["smiles"].iloc[0]
                    else:
                        row_dict["smiles_used_by_model"] = None
                else:
                    row_dict["drugbank_id_used_by_model"] = None
                    row_dict["smiles_used_by_model"] = None
            else:
                row_dict["drugbank_id_used_by_model"] = None
                row_dict["smiles_used_by_model"] = None

            rows.append(row_dict)

    df = pd.DataFrame(rows)

    os.makedirs(os.path.dirname(out_csv) if os.path.dirname(out_csv) else ".", exist_ok=True)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    df_pos = df[df["y_pred"] == 1].copy()
    df_pos.to_csv(out_pos_csv, index=False, encoding="utf-8-sig")

    n = len(df)
    n_true_pos = int((df["y_true"] == 1).sum()) if n > 0 else 0
    n_pred_pos = int((df["y_pred"] == 1).sum()) if n > 0 else 0
    print(f"[OK] wrote: {out_csv} (N={n}, true_pos={n_true_pos}, pred_pos={n_pred_pos})")
    print(f"[OK] wrote: {out_pos_csv} (N_pos={len(df_pos)})")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default="drugbank",
                        choices=["drugbank", "davis", "kiba", "biosnap", "bindingdb", "human"])
    parser.add_argument('--split', type=str, default="real_timeline",
                        choices=["cold_pair", "cold_drug", "cold_protein", "random", "cluster_start", "real_timeline"])
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=64)

    parser.add_argument('--ckpt', type=str, required=True, help="path to .pt checkpoint")

    parser.add_argument('--ablation', type=str, default=None,
                        choices=[None, "Local", "Global", "Secondary", "Tertiary", "Quaternary", "Primary"],
                        help="apply ablation only at test time (pass into model __call__ as ablation=...)")

    parser.add_argument('--out_csv', type=str, default="output/predictions/test_predictions.csv")
    parser.add_argument('--out_pos_csv', type=str, default="output/predictions/test_predictions_pos.csv")

    parser.add_argument(
        '--drugbank_map_csv',
        type=str,
        default="",
        help="CSV with columns: smiles_nid, smiles, drugbank_id"
    )

    args = parser.parse_args()

    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}")
    set_seed(args.seed)

    # Reuse Config only for preparedataset signature
    class DummyArgs:
        pass

    dummy = DummyArgs()
    dummy.batch_size = args.batch_size
    dummy.epoch = 1
    dummy.lr = 1e-5
    dummy.weight_decay = 0.0
    dummy.decay_interval = 1
    dummy.lr_decay = 1.0
    dummy.patience = 1
    dummy.min_delta = 0.0
    dummy.dataset = args.dataset
    dummy.split = args.split
    dummy.cuda = args.cuda
    dummy.seed = args.seed
    dummy.ablation = args.ablation
    config = Config(dummy)

    # Build model (must match training)
    drug_transformer = AutoModel.from_pretrained('../Model/ChemBERTa-77M-MLM')
    protein_transformer = AutoModel.from_pretrained('../Model/prot_resize')
    model = ColdDTI(384, 1024, 704, drug_transformer, protein_transformer).to(device)

    print(f"[INFO] loading ckpt: {args.ckpt}")
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state, strict=True)

    # Only testloader
    _, _, testloader, _, _, _, _ = preparedataset(
        config.batch_size, args.split, args.dataset
    )

    # Read test.csv column names (for expanding sample_row)
    sample_columns = load_test_columns(args.dataset, args.split)

    # ===== NEW: load model-used protein table =====
    # ./data/{dataset}/protein.csv has header: protein_id, protein(marked)
    protein_df_marked = pd.read_csv(
        f"./data/{args.dataset}/protein.csv"
    )
    # rename to keep code minimal and explicit
    protein_df_marked = protein_df_marked.rename(columns={
        "protein_id": "protein_id_used_by_model",
        "protein": "protein_marked"
    })

    # ===== NEW: load drugbank map =====
    drugbank_map_df = pd.read_csv(args.drugbank_map_csv)
    # hard assert to avoid silent mismatch
    need_cols = {"smiles_nid", "drugbank_id"}
    assert need_cols.issubset(set(drugbank_map_df.columns)), \
        f"drugbank_map_csv must contain columns {need_cols}, got {list(drugbank_map_df.columns)}"

    export_predictions(
        model=model,
        testloader=testloader,
        device=device,
        out_csv=args.out_csv,
        out_pos_csv=args.out_pos_csv,
        sample_columns=sample_columns,
        ablation=args.ablation,
        protein_df_marked=protein_df_marked,
        drugbank_map_df=drugbank_map_df,
    )


if __name__ == "__main__":
    main()