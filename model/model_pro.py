import torch
import torch.optim as optim
from .colddti import ColdDTI
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score,
    precision_recall_curve, auc, average_precision_score,
    accuracy_score, confusion_matrix
)
from tqdm import tqdm
import numpy as np


class Trainer(object):
    def __init__(
        self,
        model: ColdDTI,
        epoch,
        batch_size,
        lr,
        weight_decay,
        lr_decay,
        decay_interval,
        patience=10,         # from args.patience
        min_delta=0.0       # === Early Stop: 新增
    ):
        self.model = model
        self.epoch = epoch
        self.lr_decay = lr_decay
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.batch_size = batch_size
        self.decay_interval = decay_interval
        self.tester = Tester(self.model)

        # === Early Stop: 参数保存 ===
        self.patience = patience
        self.min_delta = min_delta

    def train(self, trainloader, validloader, testloader, file_model, file_AUCs, device):
        """
        训练期间仅在 validation 上评估并挑选最优；不在 test 上做选择。
        训练完成后，外部(main.py)再载入最优模型到 test 上评一次即可。
        """
        max_AUC_val = 0.0
        epoch_label = 0

        # === Early Stop: 连续不提升计数 ===
        epochs_no_improve = 0

        for epoch in range(1, self.epoch + 1):
            # 学习率衰减
            if epoch % self.decay_interval == 0:
                self.optimizer.param_groups[0]['lr'] *= self.lr_decay

            self.model.train()
            # ---- 逐 batch 累积 loss，求均值再一次性反传（避免逐样本 backward 的不稳定）----
            for item in tqdm(trainloader, desc=f"Epoch {epoch}/{self.epoch}"):
                self.optimizer.zero_grad()
                batch_losses = []

                for data in item:
                    smiles_tokenized, proteins_tokenized, smiles_content, proteins_content, interaction, _ = data
                    smiles_tokenized = smiles_tokenized.to(device)
                    proteins_tokenized = proteins_tokenized.to(device)
                    smiles_content = smiles_content.to(device)
                    proteins_content = proteins_content.to(device)
                    interaction = interaction.to(device)

                    # model(...) 返回：前向输出、loss、（可与 loss 相同）
                    _, loss, _ = self.model(
                        smiles_tokenized, proteins_tokenized,
                        smiles_content, proteins_content,
                        interaction, train=True
                    )
                    # 注意：此处 loss 是标量张量
                    batch_losses.append(loss)

                loss_batch = torch.stack(batch_losses).mean()
                loss_batch.backward()
                # 可选：梯度裁剪，稳定训练
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            # ---- 用 validation 评估并挑最优（不要用 test）----
            AUC_val, PRAUC_val, AUPRC_val, precision_val, recall_val, acc_val, *_ = self.tester.test(validloader, device)
            AUCs = [epoch, AUC_val, PRAUC_val, AUPRC_val, precision_val, recall_val, acc_val, 0, 0, 0, 0]
            self.tester.save_AUCs(AUCs, file_AUCs)
            print('\t'.join(map(str, AUCs)))

            # === Early Stop: 根据 val AUC 判断是否提升 ===
            if AUC_val > max_AUC_val + self.min_delta:
                self.tester.save_model(self.model, file_model)
                max_AUC_val = AUC_val
                epoch_label = epoch
                epochs_no_improve = 0  # reset 计数
                print(f"[VAL] AUC improved to {AUC_val:.4f}, save model.")
            else:
                epochs_no_improve += 1
                print(f"[VAL] AUC not improved for {epochs_no_improve} epoch(s).")

            # === Early Stop: 满足耐心阈值则提前停止 ===
            if self.patience is not None and epochs_no_improve >= self.patience:
                print(f"Early stopping triggered at epoch {epoch}. "
                      f"Best epoch = {epoch_label}, Best VAL AUC = {max_AUC_val:.4f}")
                break

        print("The best model (by VAL) is epoch", epoch_label)
        return epoch_label


class Tester(object):
    def __init__(self, model):
        self.model = model
        # ablation 会在外部 trainer.tester.ablation = args.ablation 时动态加上
        self.ablation = None

    def test(self, testloader, device):
        self.model.eval()
        T, Y, S = [], [], []

        with torch.no_grad():
            for item in tqdm(testloader, desc="Evaluating"):
                for data in item:
                    smiles_tokenized, proteins_tokenized, smiles_content, proteins_content, interaction, _ = data
                    smiles_tokenized = smiles_tokenized.to(device)
                    proteins_tokenized = proteins_tokenized.to(device)
                    smiles_content = smiles_content.to(device)
                    proteins_content = proteins_content.to(device)
                    interaction = interaction.to(device)

                    # eval 模式下 model(...) 返回：correct_label, predicted_label, score
                    correct_labels, predicted_labels, predicted_scores = self.model(
                        smiles_tokenized, proteins_tokenized,
                        smiles_content, proteins_content,
                        interaction, train=False,
                        ablation=self.ablation
                    )
                    T.append(correct_labels)
                    Y.append(predicted_labels)
                    S.append(predicted_scores)

        # ---- 指标计算（修正 PRAUC 计算）----
        AUC = roc_auc_score(T, S)
        precision_curve, recall_curve, _ = precision_recall_curve(T, S)
        PRAUC = auc(recall_curve, precision_curve)
        AUPRC = average_precision_score(T, S)
        precision = precision_score(T, Y)
        recall = recall_score(T, Y)
        acc = accuracy_score(T, Y)

        T = np.array(T)
        Y = np.array(Y)
        S = np.array(S)

        # 供外部记录（不是训练 loss）
        tn, fp, fn, tp = confusion_matrix(T, Y, labels=[0, 1]).ravel()

        return AUC, PRAUC, AUPRC, precision, recall, acc, S, 0.0, tn, fp, fn, tp

    def save_AUCs(self, AUCs, filename):
        with open(filename, 'a') as f:
            f.write('\t'.join(map(str, AUCs)) + '\n')

    def save_model(self, model, filename):
        torch.save(model.state_dict(), filename)


class Config(object):
    def __init__(self, args) -> None:
        self.lr = args.lr
        self.batch_size = args.batch_size
        self.weight_decay = args.weight_decay
        self.decay_interval = args.decay_interval
        self.lr_decay = args.lr_decay
        self.epoch = args.epoch

        # === Early Stop: 从 args 里读出来，传给 Trainer ===
        self.patience = args.patience
        self.min_delta = args.min_delta