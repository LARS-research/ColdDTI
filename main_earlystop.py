# -*- coding: utf-8 -*-
import os
import random
import argparse
import numpy as np
import torch
from transformers import AutoModel

from model.model_pro import Tester, Trainer, Config
from model.utils import preparedataset
# from model.colddti_selfatt_A import ColdDTI  # 你现在用的 selfatt_A
from model.colddti import ColdDTI



def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)   # ✅ 比原版更完整
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="settings")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epoch', type=int, default=100)

    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--decay_interval', type=int, default=5)
    parser.add_argument('--lr_decay', type=float, default=0.5)

    # Early Stop
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--min_delta', type=float, default=0.0)

    parser.add_argument('--dataset', type=str, default="drugbank",
                        choices=["drugbank", "davis", "kiba", "biosnap", "bindingdb", "human"])
    parser.add_argument('--split', type=str, default="cold_pair",
                        choices=["cold_pair", "cold_drug", "cold_protein", "random", "cluster_start", "real_timeline"])
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--seed', type=int, default=2025)

    # ablation：不传就是 None（不做）
    parser.add_argument(
        '--ablation',
        type=str,
        default=None,
        choices=["Local", "Global", "Secondary", "Tertiary", "Quaternary", "Primary"],
        help='ablation setting (omit to run full model)'
    )

    args = parser.parse_args()
    print(f"[INFO] dataset={args.dataset}, split={args.split}, seed={args.seed}, ablation={args.ablation}")

    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device = {device}")

    set_seed(args.seed)

    config = Config(args)

    drug_transformer = AutoModel.from_pretrained('../Model/ChemBERTa-77M-MLM')
    protein_transformer = AutoModel.from_pretrained('../Model/prot_resize')

    model = ColdDTI(384, 1024, 704, drug_transformer, protein_transformer)
    model.to(device)

    trainloader, validloader, testloader, _, _, _, _ = preparedataset(
        config.batch_size, args.split, args.dataset
    )

    # ✅ 注意：这里不把 ablation 传进 Trainer（保持公平：训练不做 ablation）
    trainer = Trainer(
        model,
        config.epoch,
        config.batch_size,
        config.lr,
        config.weight_decay,
        config.lr_decay,
        config.decay_interval,
        patience=config.patience,
        min_delta=config.min_delta
    )

    dir_result = f'output/{args.dataset}/result/'
    dir_model = f'output/{args.dataset}/model/'
    os.makedirs(dir_result, exist_ok=True)
    os.makedirs(dir_model, exist_ok=True)

    base_name = f"{args.dataset}_{args.split}_{args.seed}"
    suffix = f"_{args.ablation}" if args.ablation else ""
    file_AUCs = os.path.join(dir_result, base_name + suffix + '.txt')
    file_model = os.path.join(dir_model, base_name + suffix + '.pt')

    # ✅ header 命名更准确（训练过程评估的是 val）
    header = 'Epoch\tAUC_val\tPRAUC_val\tAUPRC_val\tPrecision_val\tRecall_val\tACC_val\tTN\tFP\tFN\tTP'
    with open(file_AUCs, 'w') as f:
        f.write(header + '\n')

    print('Training...')
    print(header)

    best_epoch = trainer.train(trainloader, validloader, testloader, file_model, file_AUCs, device)

    print("Testing best checkpoint...")
    model.load_state_dict(torch.load(file_model, map_location=device))  # ✅ 更稳

    tester = Tester(model)
    tester.ablation = args.ablation  # ✅ 只在 Tester 生效（你的默认规则）

    AUC_test, PRAUC_test, AUPRC_test, precision_test, recall_test, ACC, _, _, tn, fp, fn, tp = tester.test(testloader, device)

    results = 'BestEpoch\tAUC_test\tPRAUC\tAUPRC\tAccuracy\tPrecision\tRecall\tF1\tTN\tFP\tFN\tTP'
    f1 = 2 * precision_test * recall_test / (precision_test + recall_test + 1e-5)
    metric = [best_epoch, AUC_test, PRAUC_test, AUPRC_test, ACC, precision_test, recall_test, f1, tn, fp, fn, tp]

    with open(file_AUCs, 'a') as f:
        f.write(results + '\n')
        f.write('\t'.join(map(str, metric)) + '\n')

    print("[DONE] best_epoch =", best_epoch)
    print("[TEST]", results)
    print("[TEST]", '\t'.join(map(str, metric)))