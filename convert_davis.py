import pandas as pd
import os

# 路径
orig_csv = "../Dataset/davis/protein.csv"
mapping_csv = "./data/davis/mapping_davis_strict.csv"
out_csv = "./data/davis/protein_uniprot.csv"

# 读取文件
df_orig = pd.read_csv(orig_csv)           # Davis 原始 (protein_id=Gene Symbol, protein=Sequence)
df_map = pd.read_csv(mapping_csv)         # 映射表 (protein_symbol, uniprot_id)

# 合并：按 gene symbol 对应
df_merged = df_orig.merge(df_map, left_on="protein_id", right_on="protein_symbol", how="inner")

# 只保留 uniprot_id 和 sequence
df_final = df_merged[["uniprot_id", "protein"]].rename(columns={"uniprot_id": "protein_id"})

# 保存
os.makedirs(os.path.dirname(out_csv), exist_ok=True)
df_final.to_csv(out_csv, index=False)

print(f"Saved merged file: {out_csv}, shape={df_final.shape}")