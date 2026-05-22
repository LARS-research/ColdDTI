import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

# def extract_subseq_tensor(input_ids, last_hidden_state, start_token_id, end_token_id):
#     """
#     input_ids: Tensor of shape [batch_size, seq_len]
#     last_hidden_state: Tensor of shape [batch_size, seq_len, hidden_dim]
#     start_token_id: int, token ID for [start]
#     end_token_id: int, token ID for [end]
#     Returns:
#         Tensor of shape [num_subsequences_total, hidden_dim]
#     """
#     all_subseq_reprs = []

#     hidden_dim = last_hidden_state.size(-1)

#     input_ids_i = input_ids[0]  # [seq_len]
#     hidden_i = last_hidden_state[0]  # [seq_len, hidden_dim]
#     # 找到[start]和[end]的位置
#     start_positions = (input_ids_i == start_token_id).nonzero(as_tuple=False).squeeze(-1).tolist()
#     end_positions = (input_ids_i == end_token_id).nonzero(as_tuple=False).squeeze(-1).tolist()
#     # 一一对应匹配
#     if len(start_positions) != len(end_positions):
#         raise ValueError("Mismatched [start]/[end] counts.")
#     if any(s >= e for s, e in zip(start_positions, end_positions)):
#         raise ValueError("[start] must come before corresponding [end].")
#     for s, e in zip(start_positions, end_positions):
#         if e - s <= 1:
#             continue  # 空子序列，跳过
#         subseq_hidden = hidden_i[s:e+1]  # [len_subseq, hidden_dim]
#         subseq_mean = subseq_hidden.mean(dim=0)  # [hidden_dim]
#         all_subseq_reprs.append(subseq_mean)

#     # 合并为一个 tensor
#     if len(all_subseq_reprs) == 0:
#         return last_hidden_state.new_empty((1, 0, hidden_dim))  # 无子序列时返回空张量
#     return torch.stack(all_subseq_reprs, dim=0).unsqueeze(0)  # [bsz, num_subsequences_total, hidden_dim]
def extract_subseq_tensor(input_ids, last_hidden_state, start_token_id, end_token_id):
    """
    input_ids: Tensor [bsz, seq_len]
    last_hidden_state: Tensor [bsz, seq_len, hidden_dim]
    return: Tensor [bsz, num_subseq, hidden_dim]
    NOTE: will NEVER return empty num_subseq=0 (to avoid NaN downstream).
    """
    all_subseq_reprs = []
    hidden_dim = last_hidden_state.size(-1)

    # 假设 bsz=1（你 pipeline 里确实经常是 1；即使 >1 也先用第 0 个，不改你原逻辑）
    input_ids_i = input_ids[0]            # [seq_len]
    hidden_i = last_hidden_state[0]       # [seq_len, hidden_dim]

    start_positions = (input_ids_i == start_token_id).nonzero(as_tuple=False).squeeze(-1).tolist()
    end_positions   = (input_ids_i == end_token_id).nonzero(as_tuple=False).squeeze(-1).tolist()

    # 如果 start/end 数量不一致，直接 fallback（不要 raise，让训练继续）
    if len(start_positions) != len(end_positions):
        # fallback: 用整条 protein 的 mean 作为一个“伪子段”
        fallback = hidden_i.mean(dim=0, keepdim=True)   # [1, hidden_dim]
        return fallback.unsqueeze(0)                    # [1, 1, hidden_dim]

    # 配对遍历
    for s, e in zip(start_positions, end_positions):
        if s >= e:
            continue
        if e - s <= 1:
            continue  # 空子段，跳过

        subseq_hidden = hidden_i[s:e+1]                 # [len_subseq, hidden_dim]
        subseq_mean = subseq_hidden.mean(dim=0)         # [hidden_dim]
        all_subseq_reprs.append(subseq_mean)

    # 如果一个有效子段都没有：fallback（这是你现在 NaN 的主要来源）
    if len(all_subseq_reprs) == 0:
        fallback = hidden_i.mean(dim=0, keepdim=True)   # [1, hidden_dim]
        return fallback.unsqueeze(0)                    # [1, 1, hidden_dim]

    # 正常返回
    return torch.stack(all_subseq_reprs, dim=0).unsqueeze(0)  # [1, num_subseq, hidden_dim]
class SelfAttention(nn.Module):
    def __init__(self, dim, n_heads):
        super(SelfAttention, self).__init__()

        self.dim = dim
        self.n_heads = n_heads

        assert dim % n_heads == 0, f"dim = {dim}, while n_heads = {n_heads}"
        self.attn = nn.MultiheadAttention(dim, n_heads)
        self.fc = nn.Linear(dim, dim)
        self.ln = nn.LayerNorm(dim)

    def forward(self, x):
        hid = self.ln(self.attn(x, x, x)[0] + x)
        hid = self.ln(self.fc(hid) + hid)
        return hid

class ColdDTI(nn.Module):
    def __init__(self, smile_dim, protein_dim, dim, drug_transformer, protein_transformer,
            secondary_start_id=35, secondary_end_id=36,
            tertiary_start_id=37, tertiary_end_id=38) -> None:
        super(ColdDTI, self).__init__()
        self.smile_dim = smile_dim
        self.protein_dim = protein_dim
        self.dim = dim
        self.sqrt_dk = math.ceil(dim)

        self.secondary_start_id = secondary_start_id
        self.secondary_end_id = secondary_end_id
        self.tertiary_start_id = tertiary_start_id
        self.tertiary_end_id = tertiary_end_id

        # drug & protein feature extractor
        self.drug_extractor = drug_transformer
        self.drug_post = SelfAttention(self.smile_dim, 12)
        self.protein_extractor = protein_transformer
        self.protein_post = SelfAttention(self.protein_dim, 16)

        # smile local & protein hierarchical
        self.W_lp_s = nn.Linear(smile_dim, dim)
        self.W_lp_p = nn.Linear(protein_dim, dim)

        self.W_ls_s = nn.Linear(smile_dim, dim)
        self.W_ls_p = nn.Linear(protein_dim, dim)

        self.W_lt_s = nn.Linear(smile_dim, dim)
        self.W_lt_p = nn.Linear(protein_dim, dim)

        self.W_lq_s = nn.Linear(smile_dim, dim)
        self.W_lq_p = nn.Linear(protein_dim, dim)

        # smile global & protein hierarchical
        self.W_gp_s = nn.Linear(smile_dim, dim)
        self.W_gp_p = nn.Linear(protein_dim, dim)

        self.W_gs_s = nn.Linear(smile_dim, dim)
        self.W_gs_p = nn.Linear(protein_dim, dim)

        self.W_gt_s = nn.Linear(smile_dim, dim)
        self.W_gt_p = nn.Linear(protein_dim, dim)

        self.W_gq_s = nn.Linear(smile_dim, dim)
        self.W_gq_p = nn.Linear(protein_dim, dim)

        self.classifier = nn.Sequential(
            nn.Linear(2 * dim, 2 * dim),
            nn.LeakyReLU(),
            nn.Linear(2 * dim, dim),
            nn.LeakyReLU(),
            nn.Linear(dim, 2),
        )

    def forward(self, smile_tokenized, protein_tokenized, smile_content, protein_content, ablation=None):

        # Extract drug and protein features
        # with torch.no_grad():
            # smile_content = self.drug_extractor(smile, return_dict=True, output_hidden_states=True).hidden_states[-1] # [bsz * len * dim]
            # protein_content = self.protein_extractor(protein, return_dict=True, output_hidden_states=True).hidden_states[-1] # [bsz * len * dim]

        # Obtain Hierarchical Strcuture Representation
        
        smile_content = self.drug_post(smile_content)
        protein_content = self.protein_post(protein_content)
        
        smile_local = smile_content                                                                 # [bsz * len * dim]
        smile_global = smile_content.mean(dim = -2, keepdim=True)                                   # [bsz * dim]

        protein_primary = protein_content                                                           # [bsz * len * dim]
        protein_secondary = extract_subseq_tensor(protein_tokenized, protein_content, self.secondary_start_id, self.secondary_end_id) # [bsz * len * dim]
        protein_tertiary = extract_subseq_tensor(protein_tokenized, protein_content, self.tertiary_start_id, self.tertiary_end_id)    # [bsz * len * dim]
        protein_quarternary = protein_content.mean(dim = -2, keepdim=True)                          # [bsz * 1 * dim]

        # Smiles Local & Protein Hierarchical
        lp_map = self.W_lp_s(smile_local) @ (self.W_lp_p(protein_primary).transpose(-2, -1))        # [bsz * sl_len * pp_len]
        ls_map = self.W_ls_s(smile_local) @ (self.W_ls_p(protein_secondary).transpose(-2, -1))      # [bsz * sl_len * ps_len]
        lt_map = self.W_lt_s(smile_local) @ (self.W_lt_p(protein_tertiary).transpose(-2, -1))       # [bsz * sl_len * pt_len]
        lq_map = self.W_lq_s(smile_local) @ (self.W_lq_p(protein_quarternary).transpose(-2, -1))    # [bsz * sl_len * 1]

        # Smiles Global & Protein Hierarchical
        gp_map = self.W_gp_s(smile_global) @ (self.W_gp_p(protein_primary).transpose(-2, -1))       # [bsz * 1 * pp_len]
        gs_map = self.W_gs_s(smile_global) @ (self.W_gs_p(protein_secondary).transpose(-2, -1))     # [bsz * 1 * ps_len]
        gt_map = self.W_gt_s(smile_global) @ (self.W_gt_p(protein_tertiary).transpose(-2, -1))      # [bsz * 1 * pt_len]
        gq_map = self.W_gq_s(smile_global) @ (self.W_gq_p(protein_quarternary).transpose(-2, -1))   # [bsz * 1 * 1]

        # Smiles Global + Weighted Local
        sl_inner_logits = lp_map.mean(dim=-1) + ls_map.mean(dim=-1) + lt_map.mean(dim=-1) + lq_map.mean(dim=-1)
        sl_final = (smile_local * torch.softmax(sl_inner_logits / self.sqrt_dk, dim=-1).unsqueeze(-1)).sum(dim=-2)
        sl_logits = sl_inner_logits.mean(dim=-1, keepdim=True)
        
        sg_logits = gp_map.mean(dim=-1) + gs_map.mean(dim=-1) + gt_map.mean(dim=-1) + gq_map.mean(dim=-1)
        s_weight = torch.softmax(torch.concatenate([sl_logits, sg_logits], dim=-1), dim=-1)
        smiles_representation = s_weight[:, 0] * sl_final + s_weight[:, 1] * smile_global.squeeze(1) # [bsz * dim]

        # Protein Quarternary + Weighted Hierarchical
        pp_inner_logits = lp_map.mean(dim=-2) + gp_map.mean(dim=-2)
        pp_final = (protein_primary * torch.softmax(pp_inner_logits / self.sqrt_dk, dim=-1).unsqueeze(-1)).sum(dim=-2)
        pp_logits = pp_inner_logits.mean(dim=-1, keepdim=True)

        ps_inner_logits = ls_map.mean(dim=-2) + gs_map.mean(dim=-2)
        ps_final = (protein_secondary * torch.softmax(ps_inner_logits / self.sqrt_dk, dim=-1).unsqueeze(-1)).sum(dim=-2)
        ps_logits = ps_inner_logits.mean(dim=-1, keepdim=True)

        pt_inner_logits = lt_map.mean(dim=-2) + gt_map.mean(dim=-2)
        pt_final = (protein_tertiary * torch.softmax(pt_inner_logits / self.sqrt_dk, dim=-1).unsqueeze(-1)).sum(dim=-2)
        pt_logits = pt_inner_logits.mean(dim=-1, keepdim=True)

        pq_logits = lq_map.mean(dim=-2) + gq_map.mean(dim=-2)
        p_weight = torch.softmax(torch.concatenate([pp_logits, ps_logits, pt_logits, pq_logits], dim=-1), dim=-1)
        protein_representation = p_weight[:, 0] * pp_final + p_weight[:, 1] * ps_final + p_weight[:, 2] * pt_final + p_weight[:, 3] * protein_quarternary.squeeze(1) # [bsz * dim]
        
        # Joint Representation & Prediction
        content = torch.concatenate([smiles_representation, protein_representation], dim = -1)
        logits = self.classifier(content)
        
        
        # ============================ Explain mode: return interaction maps for visualization ======================================
        if ablation == "explain":
            return logits, {
                "lp_map": lp_map.detach(),   # Drug local × Protein primary
                "ls_map": ls_map.detach(),   # Drug local × Protein secondary
                "protein_secondary": protein_secondary.detach(),
                "smile_local": smile_local.detach(),
            }
        # ==============================================================================================================================
        return logits
    
    # def __call__(self, smiles_tokenized, proteins_tokenized, smiles_content, proteins_content, correct_interaction, train=True):
    #     Loss = nn.CrossEntropyLoss()
    #     if train:
    #         predicted_interaction = self.forward(smiles_tokenized, proteins_tokenized, smiles_content, proteins_content)
    #         class_loss = Loss(predicted_interaction, correct_interaction)
    #         return predicted_interaction, class_loss, class_loss
    #     else:
    #         predicted_interaction = self.forward(smiles_tokenized, proteins_tokenized, smiles_content, proteins_content)
    #         correct_labels = correct_interaction.to('cpu').data.numpy().item()
    #         ys = F.softmax(predicted_interaction, 1).to('cpu').data.numpy()
    #         predicted_labels = np.argmax(ys)
    #         predicted_scores = ys[0, 1]
    #         return correct_labels, predicted_labels, predicted_scores
    def __call__(
        self,
        smiles_tokenized,
        proteins_tokenized,
        smiles_content,
        proteins_content,
        correct_interaction,
        train=True,
        ablation=None,
    ):
        Loss = nn.CrossEntropyLoss()
        predicted_interaction = self.forward(
            smiles_tokenized, proteins_tokenized,
            smiles_content, proteins_content,
            ablation=ablation
        )

        if train:
            class_loss = Loss(predicted_interaction, correct_interaction)
            return predicted_interaction, class_loss, class_loss
        else:
            correct_labels = correct_interaction.to('cpu').data.numpy().item()
            ys = F.softmax(predicted_interaction, 1).to('cpu').data.numpy()
            predicted_labels = np.argmax(ys)
            predicted_scores = ys[0, 1]
            return correct_labels, predicted_labels, predicted_scores