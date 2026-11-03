"""Train ColdDTI from its frozen feature arrays.

Historical underscore options are accepted. All variants share the
validation-selected training implementation in ablations/run.py.
"""
import argparse
from pathlib import Path

from ablations.run import main as run_training, VARIANTS

ROOT = Path(__file__).resolve().parent
ABLATIONS = {"Secondary": "wo_secondary", "Tertiary": "wo_tertiary",
             "Quaternary": "wo_quaternary", "Local": "wo_local", "Global": "wo_global"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="drugbank")
    parser.add_argument("--split", default="cold_pair")
    variants = parser.add_mutually_exclusive_group()
    variants.add_argument("--variant", choices=VARIANTS)
    variants.add_argument("--ablation", choices=ABLATIONS)
    parser.add_argument("--data-root", "--data_root", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", "--output_dir", type=Path, default=ROOT / "output" / "main")
    parser.add_argument("--batch_size", "--batch-size", type=int, default=64)
    parser.add_argument("--train-samples", type=int, help="Optional training-only subset for a short real-data run")
    parser.add_argument("--epoch", "--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", "--weight-decay", type=float, default=5e-4)
    parser.add_argument("--decay_interval", "--decay-interval", type=int, default=5)
    parser.add_argument("--lr_decay", "--lr-decay", type=float, default=0.5)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min_delta", "--min-delta", type=float, default=0.0)
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--device")
    parser.add_argument("--seed", type=int, default=57)
    for name in ('secondary-start', 'secondary-end', 'tertiary-start', 'tertiary-end'):
        parser.add_argument('--' + name + '-id', type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    variant = args.variant or ABLATIONS.get(args.ablation, "Full")
    command = [
        "--dataset", args.dataset, "--split", args.split, "--variant", variant,
        "--data-root", str(args.data_root), "--output-dir", str(args.output_dir),
        "--batch-size", str(args.batch_size), "--epochs", str(args.epoch),
        "--lr", str(args.lr), "--weight-decay", str(args.weight_decay),
        "--decay-interval", str(args.decay_interval), "--lr-decay", str(args.lr_decay),
        "--patience", str(args.patience), "--min-delta", str(args.min_delta),
        "--cuda", str(args.cuda), "--seed", str(args.seed),
    ]
    if args.device:
        command.extend(["--device", args.device])
    if args.train_samples is not None:
        command.extend(["--train-samples", str(args.train_samples)])
    for name in ('secondary_start', 'secondary_end', 'tertiary_start', 'tertiary_end'):
        value = getattr(args, name + '_id')
        if value is not None:
            command.extend(['--' + name.replace('_', '-') + '-id', str(value)])
    if args.dry_run:
        command.append("--dry-run")
    return run_training(command)


if __name__ == "__main__":
    main()
