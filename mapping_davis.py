# map_to_uniprot_strict.py
import os
import time
import argparse
import requests
import pandas as pd
from tqdm import tqdm

try:
    import mygene
except ImportError as e:
    raise SystemExit("Please install mygene first: pip install mygene") from e

HEADERS = {"User-Agent": "Mozilla/5.0"}
UNIPROT_SEARCH = "https://rest.uniprot.org/uniprotkb/search"
UNIPROT_ENTRY  = "https://rest.uniprot.org/uniprotkb/{acc}.json"

def sleep(s=0.08):
    time.sleep(s)

def mygene_batch_map(symbols, species="human"):
    """
    用 mygene 批量把 symbol/alias -> UniProt 候选（Swiss-Prot/TrEMBL 都可能返回）
    返回 dict: {query_symbol: {"swissprot": [..], "trembl": [..]}}
    """
    mg = mygene.MyGeneInfo()
    res = mg.querymany(
        symbols,
        scopes="symbol,alias,name",
        fields="uniprot,symbol,name,alias",
        species=species,
        as_dataframe=False,
        returnall=False,
        verbose=False,
        size=1000,
    )
    mapping = {}
    for r in res:
        q = r.get("query")
        uni = r.get("uniprot")
        swiss, trembl = [], []
        if uni:
            if isinstance(uni, dict):
                sp = uni.get("Swiss-Prot", [])
                tr = uni.get("TrEMBL", [])
                if isinstance(sp, str): sp = [sp]
                if isinstance(tr, str): tr = [tr]
                swiss = sp or []
                trembl = tr or []
            elif isinstance(uni, str):
                swiss = [uni]
            elif isinstance(uni, list):
                swiss = uni
        mapping[q] = {"swissprot": swiss, "trembl": trembl}
    return mapping

def uniprot_entry_is_human_reviewed_and_matches(acc, raw_symbol):
    """
    用 UniProt entry 校验：
      - 物种 = 9606
      - 是否 reviewed (Swiss-Prot)
      - 基因主名/同义名是否匹配原 query（大小写/连字符不敏感）
    返回 (is_ok, score)
    """
    try:
        url = UNIPROT_ENTRY.format(acc=acc)
        r = requests.get(url, headers=HEADERS, timeout=20)
        if r.status_code != 200:
            return False, 0
        js = r.json()

        org = js.get("organism", {}).get("taxonId")
        if org != 9606:
            return False, 0

        reviewed = (js.get("entryType") == "Swiss-Prot")

        names = set()
        for g in js.get("genes", []):
            if "geneName" in g and "value" in g["geneName"]:
                names.add(g["geneName"]["value"])
            for syn in g.get("synonyms", []):
                if "value" in syn:
                    names.add(syn["value"])

        norm = lambda s: s.lower().replace("_", "").replace("-", "")
        q = norm(raw_symbol)
        match = any(norm(n) == q for n in names) or any(q in norm(n) for n in names)

        score = (2 if reviewed else 0) + (1 if match else 0)
        return True, score
    except requests.RequestException:
        return False, 0

def fallback_uniprot_search_by_symbol(symbol):
    """
    mygene 无结果时，用 UniProt 官方搜索回退（优先 reviewed + human）
    """
    filters = [
        f"(gene_exact:{symbol}) AND organism_id:9606 AND reviewed:true",
        f"(gene:{symbol}) AND organism_id:9606 AND reviewed:true",
        f"(protein_name:{symbol}) AND organism_id:9606 AND reviewed:true",
        f"(gene_exact:{symbol}) AND organism_id:9606",
        f"(gene:{symbol}) AND organism_id:9606",
        f"(protein_name:{symbol}) AND organism_id:9606",
    ]
    for q in filters:
        params = {
            "query": q,
            "fields": "accession,protein_name,gene_primary,organism_id,reviewed",
            "format": "tsv",
            "size": "1",
        }
        try:
            r = requests.get(UNIPROT_SEARCH, params=params, headers=HEADERS, timeout=20)
            if r.status_code == 200:
                lines = r.text.strip().split("\n")
                if len(lines) > 1:
                    acc = lines[1].split("\t")[0]
                    sleep()
                    return acc
        except requests.RequestException:
            pass
        sleep()
    return None

def choose_best_candidate(candidates, symbol):
    best_acc, best_score = None, -1
    for acc in candidates:
        ok, score = uniprot_entry_is_human_reviewed_and_matches(acc, symbol)
        if ok and score > best_score:
            best_acc, best_score = acc, score
        sleep()
    return best_acc

def main():
    ap = argparse.ArgumentParser(description="Map gene symbols/aliases to UniProt with strict UniProt validation")
    ap.add_argument("--input_csv", type=str, default="../Dataset/davis/protein.csv",
                    help="原始 CSV，包含 protein_id（基因符号/别名） 和 protein（序列）")
    ap.add_argument("--out_dir", type=str, default="./data/davis",
                    help="输出目录：mapping 与 protein_uniprot.csv 会写到这里")
    ap.add_argument("--species", type=str, default="human", help="mygene 物种过滤")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.read_csv(args.input_csv)
    if not {"protein_id", "protein"}.issubset(df.columns):
        raise ValueError("input_csv 必须包含列: protein_id, protein")

    symbols = df["protein_id"].astype(str).tolist()

    # 1) mygene 批量候选
    mg_map = mygene_batch_map(symbols, species=args.species)

    mapped, unmapped = {}, []
    for sym in tqdm(symbols, desc="Mapping (mygene + UniProt strict)"):
        cands = []
        m = mg_map.get(sym, {})
        cands.extend(m.get("swissprot", []))
        cands.extend(m.get("trembl", []))
        cands = list(dict.fromkeys(cands))  # 去重保序

        acc = choose_best_candidate(cands, sym) if cands else None
        if not acc:
            acc = fallback_uniprot_search_by_symbol(sym)

        if acc:
            ok, _ = uniprot_entry_is_human_reviewed_and_matches(acc, sym)
            if ok:
                mapped[sym] = acc
            else:
                unmapped.append(sym)
        else:
            unmapped.append(sym)

    # 2) 保存映射表
    map_df = pd.DataFrame(
        {"protein_symbol": list(mapped.keys()),
         "uniprot_id": [mapped[k] for k in mapped.keys()]}
    )
    map_path = os.path.join(args.out_dir, "mapping_davis_strict.csv")
    map_df.to_csv(map_path, index=False)

    if unmapped:
        with open(os.path.join(args.out_dir, "mapping_unmapped.log"), "w") as f:
            for s in unmapped:
                f.write(s + "\n")

    # 3) 合并原始序列，生成 protein_uniprot.csv（带序列；不覆盖旧文件）
    merged = df.merge(map_df, left_on="protein_id", right_on="protein_symbol", how="inner")
    new_df = merged[["uniprot_id", "protein"]].rename(columns={"uniprot_id": "protein_id"})
    new_df = new_df.drop_duplicates(subset=["protein_id"])  # 以 UniProt 去重
    out_path = os.path.join(args.out_dir, "protein_uniprot.csv")
    new_df.to_csv(out_path, index=False)

    print(f"[Summary] mapped={len(mapped)}, unmapped={len(unmapped)}")
    print(f"Saved: {map_path}")
    if unmapped:
        print(f"Saved: {os.path.join(args.out_dir, 'mapping_unmapped.log')}")
    print(f"Saved: {out_path} (protein_id=UniProt, protein=原始序列, n={len(new_df)})")

if __name__ == "__main__":
    main()