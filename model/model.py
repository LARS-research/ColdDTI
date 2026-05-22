import torch
import torch.optim as optim
from .colddti import ColdDTI
from sklearn.metrics import roc_auc_score, precision_score, recall_score, precision_recall_curve, auc, average_precision_score,accuracy_score, confusion_matrix
from tqdm import tqdm
import timeit
import numpy as np


class Trainer(object):
    def __init__(self, model: ColdDTI, epoch, batch_size, lr, weight_decay, lr_decay, decay_interval):
        self.model = model
        self.epoch = epoch
        self.lr_decay = lr_decay
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=lr, weight_decay=weight_decay)
        self.batch_size = batch_size
        self.decay_interval = decay_interval
        self.tester = Tester(self.model)

    def train(self, trainloader, validloader, testloader, file_model, file_AUCs, device):
        max_AUC_test = 0
        epoch_label = 0

        for epoch in range(1, self.epoch+1):
            if epoch % self.decay_interval == 0:
                self.optimizer.param_groups[0]['lr'] *= self.lr_decay
            
            self.model.train()
            loss = 0.0
            for item in tqdm(trainloader):
                train_loss = 0.0
                for data in item:
                    smiles_tokenized, proteins_tokenized, smiles_content, proteins_content, interaction, _ = data
                    smiles_tokenized = smiles_tokenized.to(device)
                    proteins_tokenized = proteins_tokenized.to(device)
                    smiles_content = smiles_content.to(device)
                    proteins_content = proteins_content.to(device)
                    interaction = interaction.to(device)
                    
                    _, loss, class_loss = self.model(smiles_tokenized, proteins_tokenized, smiles_content, proteins_content, interaction)
                    train_loss += class_loss.item()
                    loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
            # return loss
            # AUC_dev, PRAUC_dev, AUPRC_dev, precision_dev, recall_dev, acc_dev, _, _ = self.tester.test(validloader, device)
            # AUC_test, PRAUC_test, AUPRC_test, precision_test, recall_test, acc_test, _, _ = self.tester.test(testloader, device)
            # AUCs = [epoch, AUC_test, PRAUC_test, AUPRC_test, precision_test, recall_test, acc_test]
            AUC_test, PRAUC_test, AUPRC_test, precision_test, recall_test, acc_test, _, _, tn, fp, fn, tp = self.tester.test(testloader, device)
            AUCs = [epoch, AUC_test, PRAUC_test, AUPRC_test, precision_test, recall_test, acc_test, tn, fp, fn, tp]
            if AUC_test > max_AUC_test:
                self.tester.save_model(self.model, file_model)
                max_AUC_test = AUC_test
                epoch_label = epoch
            self.tester.save_AUCs(AUCs, file_AUCs)
            print('\t'.join(map(str, AUCs)))
        print("The best model is epoch", epoch_label)
        return epoch_label

class Tester(object):
    def __init__(self, model):
        self.model = model

    def test(self, testloader, device):
        self.model.eval()
        T, Y, S= [], [], []
        for item in tqdm(testloader):
            for data in item:
                smiles_tokenized, proteins_tokenized, smiles_content, proteins_content, interaction, _ = data
                smiles_tokenized = smiles_tokenized.to(device)
                proteins_tokenized = proteins_tokenized.to(device)
                smiles_content = smiles_content.to(device)
                proteins_content = proteins_content.to(device)
                interaction = interaction.to(device)

                correct_labels, predicted_labels, predicted_scores = self.model(smiles_tokenized, proteins_tokenized, smiles_content, proteins_content, interaction, train=False)
                T.append(correct_labels)
                Y.append(predicted_labels)
                S.append(predicted_scores)
        
        AUC = roc_auc_score(T, S)
        tpr, fpr, _ = precision_recall_curve(T, S)
        PRAUC = auc(fpr,tpr)
        AUPRC = average_precision_score(T,S)
        precision = precision_score(T, Y)
        recall = recall_score(T, Y)
        acc = accuracy_score(T, Y)

        T = np.array(T)
        S = np.array(S)
        loss = -np.mean(T * np.log(S) + (1 - T) * np.log(1 - S))
        tn, fp, fn, tp = confusion_matrix(T, Y, labels=[0, 1]).ravel()

        return AUC, PRAUC, AUPRC, precision, recall, acc, S, loss, tn, fp, fn, tp

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