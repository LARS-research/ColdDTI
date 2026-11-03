"""Deterministic secondary-boundary controls on residue positions.

Spans use Python's half-open [start, end) convention. Returned units contain
indices into the original encoded sequence; cached embeddings never move.
"""
import random
from model.segments import marker_spans


def _gaps_from_spans(spans, size):
    previous = 0
    gaps = []
    for start, end in spans:
        if start < previous:
            raise ValueError("Secondary segments overlap; cannot preserve their lengths")
        gaps.append(start - previous)
        previous = end
    return gaps + [size - previous]


def _place(lengths, gaps):
    spans, cursor = [], gaps[0]
    for i, length in enumerate(lengths):
        spans.append((cursor, cursor + length))
        cursor += length + gaps[i + 1]
    return spans


def secondary_units(tokens, mode="annotated", seed=57, protein_key=0,
                    secondary_start=35, secondary_end=36,
                    tertiary_start=37, tertiary_end=38,
                    residue_min=5, residue_max=29):
    """Keep segment count and residue lengths within each tertiary scope.

    random: uniformly sample a nonnegative allocation of the remaining gaps.
    shuffle: permute annotated segment lengths and the original gap lengths.
    Both change segment membership, rather than only permuting pooled units.
    Controls are fixed per protein, independent of batch order and epoch.
    """
    if mode not in ("annotated", "random", "shuffle"):
        raise ValueError(f"Unknown boundary mode: {mode}")
    rng = random.Random(seed + int(protein_key))
    annotated = marker_spans(tokens, secondary_start, secondary_end)
    annotated_scopes = marker_spans(tokens, tertiary_start, tertiary_end)
    scopes, cursor = [], 0
    for left, right in annotated_scopes:
        if cursor < left:
            scopes.append((cursor, left))
        scopes.append((left, right))
        cursor = right
    if cursor < len(tokens):
        scopes.append((cursor, len(tokens)))
    # Partial domain annotations leave real secondary units outside domain
    # markers. Keep those units in separate gap scopes instead of dropping them.
    units = []
    for left, right in scopes:
        residues = [i for i in range(left, right) if residue_min <= tokens[i] <= residue_max]
        residue_index = {position: i for i, position in enumerate(residues)}
        spans = []
        for start, end in annotated:
            if left <= start and end <= right:
                members = [residue_index[i] for i in range(start, end) if i in residue_index]
                if members:
                    spans.append((members[0], members[-1] + 1))
        if not spans:
            continue
        lengths = [end - start for start, end in spans]
        gaps = _gaps_from_spans(spans, len(residues))
        if mode == "random":
            spare = sum(gaps)
            # Stars and bars: each gap allocation has equal probability.
            dividers = sorted(rng.sample(range(spare + len(lengths)), len(lengths)))
            boundaries = [-1] + dividers + [spare + len(lengths)]
            gaps = [b - a - 1 for a, b in zip(boundaries, boundaries[1:])]
            spans = _place(lengths, gaps)
        elif mode == "shuffle":
            rng.shuffle(lengths)
            rng.shuffle(gaps)
            spans = _place(lengths, gaps)
        units.extend([residues[start:end] for start, end in spans])
    return units
