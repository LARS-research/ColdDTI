"""One architecture shared by all paper ablations.

The full variant reuses the released model's projections, post-processing and
score scale. Ablated branches are removed from both interaction and fusion.
"""
import torch
from torch import nn

from model.colddti import ColdDTI as ReleasedColdDTI
from .boundaries import marker_spans, secondary_units


VARIANTS = ("Full", "wo_secondary", "wo_tertiary", "wo_quaternary",
            "wo_local", "wo_global", "self_attention",
            "annotated_secondary", "random_secondary", "shuffle_secondary")


class WithinUnitAttention(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.linear = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, unit):
        unit = self.norm1(unit + self.attention(unit, unit, unit, need_weights=False)[0])
        return self.norm2(unit + self.linear(unit))


class ColdDTIAblation(ReleasedColdDTI):
    # The old model overrides __call__ for a trainer-specific loss interface.
    # Use the standard PyTorch call path here; the runner computes the loss.
    __call__ = nn.Module.__call__

    def __init__(self, variant="Full", smile_dim=384, protein_dim=1024, dim=704,
                 boundary_seed=57, secondary_start_id=35, secondary_end_id=36,
                 tertiary_start_id=37, tertiary_end_id=38):
        if variant not in VARIANTS:
            raise ValueError(f"Unknown ablation: {variant}")
        # Encoders have already produced the cached inputs; do not load their
        # unused weights again or fine-tune them in an ablation.
        super().__init__(smile_dim, protein_dim, dim, None, None,
                         secondary_start_id, secondary_end_id, tertiary_start_id, tertiary_end_id)
        self.variant = variant
        self.boundary_seed = boundary_seed
        self.control_cache = {}
        self.control_counts = {"proteins": 0, "without_secondary": 0, "changed": 0}
        if variant == "self_attention":
            self.secondary_attention = WithinUnitAttention(protein_dim)
            self.tertiary_attention = WithinUnitAttention(protein_dim)

    def _pool(self, tokens, hidden, start, end, attention=None):
        spans = marker_spans(tokens, start, end)
        pooled = []
        for left, right in spans:
            # Preserve the released model's inclusive marker pooling.
            unit = hidden[:, left - 1:right + 1]
            if attention is not None:
                unit = attention(unit)
            pooled.append(unit.mean(dim=1))
        return torch.stack(pooled, dim=1) if pooled else hidden.mean(dim=1, keepdim=True)

    def _secondary_control(self, tokens, hidden, protein_key):
        key = int(protein_key) if protein_key is not None else tuple(tokens)
        if key not in self.control_cache:
            kwargs = dict(seed=self.boundary_seed, protein_key=protein_key or 0,
                          secondary_start=self.secondary_start_id, secondary_end=self.secondary_end_id,
                          tertiary_start=self.tertiary_start_id, tertiary_end=self.tertiary_end_id)
            annotated = secondary_units(tokens, "annotated", **kwargs)
            mode = self.variant.split("_")[0]
            units = secondary_units(tokens, mode, **kwargs)
            self.control_cache[key] = units
            self.control_counts["proteins"] += 1
            self.control_counts["without_secondary"] += int(not annotated)
            self.control_counts["changed"] += int(units != annotated)
        units = self.control_cache[key]
        if not units:
            return hidden.mean(dim=1, keepdim=True)
        return torch.stack([hidden[:, indices].mean(dim=1) for indices in units], dim=1)

    def forward(self, smile_tokenized, protein_tokenized, smile_content, protein_content,
                protein_key=None, return_maps=False, return_repr=False):
        if smile_content.shape[0] != 1 or protein_content.shape[0] != 1:
            raise ValueError("Pass one variable-length pair at a time; the runner accumulates batches")
        smile = self.drug_post(smile_content)
        protein = self.protein_post(protein_content)
        tokens = protein_tokenized.reshape(-1).tolist()
        drug = {"l": smile, "g": smile.mean(dim=1, keepdim=True)}
        protein_levels = {"p": protein, "q": protein.mean(dim=1, keepdim=True)}
        if self.variant != "wo_secondary":
            if self.variant.endswith("_secondary") and self.variant != "wo_secondary":
                secondary = self._secondary_control(tokens, protein, protein_key)
            else:
                attention = self.secondary_attention if self.variant == "self_attention" else None
                secondary = self._pool(tokens, protein, self.secondary_start_id, self.secondary_end_id, attention)
            protein_levels["s"] = secondary
        if self.variant != "wo_tertiary":
            attention = self.tertiary_attention if self.variant == "self_attention" else None
            protein_levels["t"] = self._pool(tokens, protein, self.tertiary_start_id, self.tertiary_end_id, attention)
        if self.variant == "wo_quaternary":
            del protein_levels["q"]
        if self.variant == "wo_local":
            del drug["l"]
        if self.variant == "wo_global":
            del drug["g"]

        # Use the original level order for the full-model comparison.
        protein_levels = {key: protein_levels[key] for key in ("p", "s", "t", "q") if key in protein_levels}
        maps = {}
        for d, x in drug.items():
            for p, y in protein_levels.items():
                maps[d, p] = (getattr(self, f"W_{d}{p}_s")(x)
                              @ getattr(self, f"W_{d}{p}_p")(y).transpose(-2, -1))

        drug_scores = {d: sum(maps[d, p].mean(dim=-1) for p in protein_levels) for d in drug}
        protein_scores = {p: sum(maps[d, p].mean(dim=-2) for d in drug) for p in protein_levels}
        drug_representation = self._fuse(drug, drug_scores)
        protein_representation = self._fuse(protein_levels, protein_scores)
        representation = torch.cat([drug_representation, protein_representation], dim=-1)
        logits = self.classifier(representation)
        if return_maps:
            return logits, dict(interaction_maps={d + p: value for (d, p), value in maps.items()},
                                drug_scores=drug_scores, protein_scores=protein_scores)
        if return_repr:
            return logits, representation
        return logits

    def _fuse(self, levels, scores):
        representations, logits = [], []
        for level, features in levels.items():
            attention = torch.softmax(scores[level] / self.sqrt_dk, dim=-1).unsqueeze(-1)
            representations.append((features * attention).sum(dim=1))
            logits.append(scores[level].mean(dim=-1, keepdim=True))
        weights = torch.softmax(torch.cat(logits, dim=-1), dim=-1)
        return sum(weights[:, i:i + 1] * value for i, value in enumerate(representations))
