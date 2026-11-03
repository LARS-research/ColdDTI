"""Export predictions from a trained ColdDTI checkpoint and its frozen features."""
import argparse
import json
from pathlib import Path

from ablations.run import VARIANTS
from main import ABLATIONS

ROOT = Path(__file__).resolve().parent


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--dataset")
    parser.add_argument("--split")
    parser.add_argument("--data-root", "--data_root", type=Path)
    variants = parser.add_mutually_exclusive_group()
    variants.add_argument("--variant", choices=VARIANTS)
    variants.add_argument("--ablation", choices=ABLATIONS)
    parser.add_argument("--batch_size", "--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=57)
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--device")
    parser.add_argument("--out_csv", type=Path, default=ROOT / "output/predictions/test_predictions.csv")
    parser.add_argument("--out_pos_csv", type=Path, default=ROOT / "output/predictions/test_predictions_pos.csv")
    parser.add_argument("--drugbank_map_csv", type=Path)
    args = parser.parse_args(argv)

    import pandas as pd
    import torch
    from torch.utils.data import DataLoader
    from ablations.data import InteractionDataset, keep_variable_lengths
    from ablations.run import evaluate, set_seed
    from model.checkpoint import load_checkpoint

    device = torch.device(args.device or (f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu"))
    set_seed(args.seed)
    model, features, config = load_checkpoint(args.ckpt, args.data_root, device,
        dataset=args.dataset, split=args.split, variant=args.variant or ABLATIONS.get(args.ablation))
    dataset, split, variant = config['dataset'], config['split'], config['variant']
    directory = features.directory
    test = InteractionDataset(directory / split / "test.csv", features)
    loader = DataLoader(test, batch_size=args.batch_size, shuffle=False, collate_fn=keep_variable_lengths)
    metrics, frame = evaluate(model, loader, device)
    frame["y_true"] = frame.Y.astype(int)
    frame["y_pred"] = frame.prediction
    frame["y_score"] = frame.score
    protein_table = directory / "protein.csv"
    if protein_table.exists():
        proteins = pd.read_csv(protein_table)
        frame["protein_id_used_by_model"] = proteins.iloc[frame.protein_nid].protein_id.to_numpy()
    if args.drugbank_map_csv:
        mapping = pd.read_csv(args.drugbank_map_csv).drop_duplicates("smiles_nid")
        frame["drugbank_id_used_by_model"] = frame.smiles_nid.map(mapping.set_index("smiles_nid").drugbank_id)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    args.out_pos_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out_csv, index=False)
    frame.loc[frame.y_pred.eq(1)].to_csv(args.out_pos_csv, index=False)
    print(json.dumps(dict(dataset=dataset, split=split, variant=variant, samples=len(frame),
                          checkpoint=str(args.ckpt), **metrics), indent=2))
    print(f"Saved {args.out_csv} and {args.out_pos_csv}")


if __name__ == "__main__":
    main()
