"""Nonoverlapping marker scopes shared by main and ablation models."""


def marker_spans(tokens, start_id, end_id):
    starts = [i for i, token in enumerate(tokens) if token == start_id]
    ends = [i for i, token in enumerate(tokens) if token == end_id]
    if len(starts) != len(ends):
        raise ValueError(f'Unpaired hierarchy tokens: {start_id}/{end_id}')
    if any(s + 1 >= e for s, e in zip(starts, ends)):
        raise ValueError(f'Empty or reversed hierarchy scope: {start_id}/{end_id}')
    if any(right_start < left_end for left_end, right_start in zip(ends, starts[1:])):
        raise ValueError(f'Overlapping hierarchy scopes: {start_id}/{end_id}')
    return [(start + 1, end) for start, end in zip(starts, ends)]
