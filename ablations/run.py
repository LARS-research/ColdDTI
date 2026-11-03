#!/usr/bin/env python3
"""Train the paper ablations with validation-only checkpoint selection."""
import argparse
import json
import random
from pathlib import Path
import sys
import time
import uuid


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
VARIANTS = ("Full", "wo_secondary", "wo_tertiary", "wo_quaternary", "wo_local", "wo_global",
            "self_attention", "annotated_secondary", "random_secondary", "shuffle_secondary")
SUITES = {
    "protein": VARIANTS[:4],
    "drug": ("Full", "wo_local", "wo_global"),
    "attention": ("Full", "self_attention"),
    "boundary": ("annotated_secondary", "random_secondary", "shuffle_secondary"),
    "paper": VARIANTS,
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group()
    choice.add_argument("--variants", "--variant", nargs="+", choices=VARIANTS)
    choice.add_argument("--suite", choices=SUITES)
    parser.add_argument("--dataset", default="drugbank")
    parser.add_argument("--splits", "--split", nargs="+", default=["cold_pair"],
                        choices=["cold_drug", "cold_protein", "cold_pair", "random", "real_timeline", "cluster", "cluster_start"])
    parser.add_argument("--seeds", "--seed", nargs="+", type=int, default=[57])
    parser.add_argument("--boundary-seed", type=int, default=57)
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "output" / "ablations")
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--device", help="Override device, e.g. cpu or cuda:1")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--train-samples", type=int, help="Optional fixed real training subset; val/test remain complete")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--decay-interval", type=int, default=5)
    parser.add_argument("--lr-decay", type=float, default=0.5)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--smile-dim", type=int, default=384)
    parser.add_argument("--protein-dim", type=int, default=1024)
    parser.add_argument("--dim", type=int, default=704)
    parser.add_argument("--secondary-start-id", type=int)
    parser.add_argument("--secondary-end-id", type=int)
    parser.add_argument("--tertiary-start-id", type=int)
    parser.add_argument("--tertiary-end-id", type=int)
    parser.add_argument("--dry-run", action="store_true", help="List runs without loading features or PyTorch")
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be positive")
    if args.train_samples is not None and args.train_samples < 1:
        parser.error("--train-samples must be positive")
    args.variants = list(SUITES[args.suite]) if args.suite else (args.variants or ["Full"])
    return args


def set_seed(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def evaluate(model, loader, device):
    import numpy as np
    import torch
    from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score, log_loss,
                                 precision_score, recall_score, accuracy_score)
    labels, scores, indices = [], [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            for features, label, protein, index in batch:
                logits = model(*(value.to(device) for value in features), protein_key=protein)
                labels.append(label)
                scores.append(logits.softmax(dim=-1)[0, 1].item())
                indices.append(index)
    predictions = np.asarray(scores) >= 0.5
    metrics = {
        "auc": float(roc_auc_score(labels, scores)),
        "aupr": float(average_precision_score(labels, scores)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "log_loss": float(log_loss(labels, scores, labels=[0, 1])),
    }
    frame = loader.dataset.frame.iloc[indices].copy()
    frame["score"] = scores
    frame["prediction"] = predictions.astype(int)
    return metrics, frame


def train_one(args, split, variant, seed, features, device):
    import torch
    import pandas as pd
    from ablations.data import make_loaders
    from ablations.model import ColdDTIAblation
    from model.checkpoint import make_contract, save_checkpoint, checkpoint_state
    from preprocess.artifacts import atomic_json, dataset_lock

    # New caches carry their actual tokenizer IDs. Legacy caches retain the
    # historical 35/36/37/38 defaults; explicit CLI values override metadata.
    for name, default in zip(('secondary_start', 'secondary_end', 'tertiary_start', 'tertiary_end'),
                             (35, 36, 37, 38)):
        key = name + '_id'
        if getattr(args, key) is None:
            setattr(args, key, features.metadata.get('tag_ids', {}).get(name, default))
    set_seed(seed)
    model = ColdDTIAblation(
        variant, args.smile_dim, args.protein_dim, args.dim, args.boundary_seed,
        args.secondary_start_id, args.secondary_end_id, args.tertiary_start_id, args.tertiary_end_id,
    ).to(device)
    directory = args.output_dir / args.dataset / split / variant / f"seed_{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    configuration = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    configuration.update(variant=variant, split=split, seed=seed,
                         run_id=uuid.uuid4().hex,
                         token_attention_batch_first=model.drug_post.attn.batch_first,
                         preprocessing=features.preprocessing, feature_metadata=features.metadata)
    with dataset_lock(features.directory):
        train, val, test = make_loaders(args.data_root / args.dataset, split, features, args.batch_size, seed,
                                      train_samples=args.train_samples)
        contract = make_contract(configuration, features)
        for part, loader in (('train', train), ('val', val), ('test', test)):
            if loader.dataset.source_sha256 != contract['data'][split + '/' + part + '.csv']:
                raise RuntimeError('Split changed while preparing the training snapshot')
    configuration['artifact_contract'] = contract
    atomic_json(directory / 'config.json', configuration)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    started = time.monotonic()
    initial_weight = model.classifier[0].weight.detach().cpu().clone()
    initial_validation, _ = evaluate(model, val, device)
    (directory / "initial_validation.json").write_text(json.dumps(initial_validation, indent=2) + "\n")
    print(f"INITIAL {split} {variant} seed={seed} val_auc={initial_validation['auc']:.6f} "
          f"val_log_loss={initial_validation['log_loss']:.6f}", flush=True)
    first_gradient_norm = None
    best_auc, best_epoch, stalled = float("-inf"), 0, 0
    history = []
    for epoch in range(1, args.epochs + 1):
        if args.decay_interval > 0 and epoch % args.decay_interval == 0:
            for group in optimizer.param_groups:
                group["lr"] *= args.lr_decay
        model.train()
        total_loss = 0.0
        epoch_started = time.monotonic()
        for batch_index, batch in enumerate(train):
            optimizer.zero_grad()
            # Mean batch gradient without retaining every pair's graph.
            for values, label, protein, _ in batch:
                logits = model(*(value.to(device) for value in values), protein_key=protein)
                loss = torch.nn.functional.cross_entropy(logits, torch.tensor([label], device=device))
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite training loss: {split}, {variant}, epoch={epoch}")
                total_loss += loss.item()
                (loss / len(batch)).backward()
            if epoch == 1 and batch_index == 0:
                # Measure, without clipping, the actual initial batch gradient.
                first_gradient_norm = float(torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float("inf"), error_if_nonfinite=True).item())
            optimizer.step()
        validation, _ = evaluate(model, val, device)
        history.append({"epoch": epoch, "train_loss": total_loss / len(train.dataset),
                        **{f"val_{key}": value for key, value in validation.items()}})
        pd.DataFrame(history).to_csv(directory / "validation.csv", index=False)
        if validation["auc"] > best_auc + args.min_delta:
            best_auc, best_epoch, stalled = validation["auc"], epoch, 0
            save_checkpoint(directory / 'best.pt', model, contract)
        else:
            stalled += 1
        print(f"{split} {variant} seed={seed} epoch={epoch} "
              f"train_loss={history[-1]['train_loss']:.6f} val_auc={validation['auc']:.6f} "
              f"val_log_loss={validation['log_loss']:.6f} seconds={time.monotonic()-epoch_started:.1f}", flush=True)
        if args.patience > 0 and stalled >= args.patience:
            break

    model.load_state_dict(checkpoint_state(directory / 'best.pt', contract, device))
    # The test set is evaluated once, after validation selects the checkpoint.
    metrics, predictions = evaluate(model, test, device)
    result = dict(dataset=args.dataset, split=split, variant=variant, seed=seed,
                  boundary_seed=args.boundary_seed, best_epoch=best_epoch,
                  best_val_auc=best_auc, initial_val_auc=initial_validation["auc"],
                  first_gradient_norm=first_gradient_norm,
                  classifier_weight_delta_l2=float((model.classifier[0].weight.detach().cpu()-initial_weight).norm().item()),
                  elapsed_seconds=time.monotonic()-started,
                  train_samples=len(train.dataset), val_samples=len(val.dataset), test_samples=len(test.dataset),
                  device=str(device), token_attention_batch_first=model.drug_post.attn.batch_first, **metrics)
    (directory / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    (directory / "boundary_counts.json").write_text(json.dumps(model.control_counts, indent=2) + "\n")
    predictions.to_csv(directory / "test_predictions.csv", index=False)
    print("TEST " + json.dumps(result), flush=True)
    return result


def main(argv=None):
    args = parse_args(argv)
    jobs = [(split, variant, seed) for split in args.splits for variant in args.variants for seed in args.seeds]
    if args.dry_run:
        for split, variant, seed in jobs:
            print(f"{args.dataset} {split} {variant} seed={seed} boundary_seed={args.boundary_seed}")
        print(f"Total runs: {len(jobs)}")
        return
    import torch
    from ablations.data import FeatureStore
    from ablations.summarize import summarize
    device = torch.device(args.device or (f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu"))
    features = FeatureStore(args.data_root / args.dataset)
    for split, variant, seed in jobs:
        train_one(args, split, variant, seed, features, device)
        summarize(args.output_dir)


if __name__ == "__main__":
    main()
