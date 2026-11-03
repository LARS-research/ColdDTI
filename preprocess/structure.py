"""Read residue-aligned mmCIF annotations and render hierarchy tokens.

Default tertiary scopes are chains. Domain mode requires actual
_pdbx_domain_range records or an explicit residue-range CSV.
"""
from collections import defaultdict
from pathlib import Path

from Bio.PDB import FastMMCIFParser
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from Bio.PDB.Polypeptide import is_aa
from Bio.SeqUtils import seq1


def as_list(value):
    return [] if value is None else value if isinstance(value, list) else [value]


def ranges_from_category(data, category, types=None, forced_type=None):
    prefix = '_' + category + '.'
    keys = ('beg_label_asym_id', 'beg_label_seq_id', 'end_label_asym_id', 'end_label_seq_id')
    columns = [as_list(data.get(prefix + key)) for key in keys]
    if not any(columns):
        return []
    if len({len(column) for column in columns}) != 1:
        raise ValueError('Inconsistent annotation columns in ' + category)
    size = len(columns[0])
    types = as_list(types) if types is not None else [forced_type] * size
    if len(types) != size:
        raise ValueError('Inconsistent annotation types in ' + category)
    result = []
    for left_chain, left, right_chain, right, kind in zip(*columns, types):
        if left_chain != right_chain:
            raise ValueError('A tagged range cannot cross chains')
        if str(left) in ('.', '?') or str(right) in ('.', '?'):
            continue
        left, right = int(left), int(right)
        if left < 1 or right < left:
            raise ValueError('Invalid residue annotation range')
        result.append(dict(chain_id=left_chain, start=left, end=right, type=kind))
    return result


def ss_type_to_char(kind):
    kind = str(kind).upper()
    if 'HELX' in kind:
        return 'H'
    if 'SHEET' in kind or 'STRN' in kind:
        return 'E'
    if 'TURN' in kind:
        return 'T'
    if 'BEND' in kind:
        return 'B'
    return 'C'


def extract_secondary_structure_blocks(cif_file):
    return secondary_blocks(MMCIF2Dict(str(cif_file)))


def secondary_blocks(data):
    rows = ranges_from_category(data, 'struct_conf', data.get('_struct_conf.conf_type_id'))
    rows += ranges_from_category(data, 'struct_sheet_range', forced_type='SHEET')
    result, seen = defaultdict(list), set()
    for row in rows:
        row['type'] = ss_type_to_char(row['type'])
        key = row['chain_id'], row['start'], row['end'], row['type']
        if key not in seen:
            seen.add(key)
            result[row['chain_id']].append(row)
    return result


def load_structure(cif_file):
    return FastMMCIFParser(QUIET=True, auth_chains=False, auth_residues=False).get_structure(
        'protein', str(cif_file))


def extract_sequences(structure, model_index=0):
    model = list(structure.get_models())[model_index]
    sequences, residue_ids = {}, {}
    for chain in model:
        residues = [residue for residue in chain if is_aa(residue, standard=False)]
        if residues:
            sequences[chain.id] = ''.join(seq1(residue.resname, custom_map={'MSE': 'M', 'UNK': 'X'})
                                          for residue in residues)
            residue_ids[chain.id] = [int(residue.id[1]) for residue in residues]
    return sequences, residue_ids


def _positions(row, residue_ids):
    indices = [i for i, number in enumerate(residue_ids) if row['start'] <= number <= row['end']]
    return (indices[0], indices[-1] + 1) if indices else None


def _secondary_units(blocks, residue_ids):
    # Resolve overlaps by H > E > T > B > C. Adjacent original records
    # remain distinct; no residue can be pooled into two secondary units.
    priorities = {'H': 0, 'E': 1, 'T': 2, 'B': 3, 'C': 4}
    owner = [None] * len(residue_ids)
    kinds = {}
    ordered = sorted(blocks, key=lambda row: (priorities[row['type']], row['start'], row['end']))
    for key, row in enumerate(ordered):
        kinds[key] = row['type']
        for i, number in enumerate(residue_ids):
            if row['start'] <= number <= row['end'] and owner[i] is None:
                owner[i] = key
    units, start = [], 0
    while start < len(owner):
        end = start + 1
        while end < len(owner) and owner[end] == owner[start]:
            end += 1
        if owner[start] is not None:
            units.append(dict(start=start, end=end, type=kinds[owner[start]]))
        start = end
    return units


def extract_hierarchical(cif_file, tertiary_mode='chain', domain_ranges=None,
                         chains=None, model_index=0):
    if tertiary_mode not in ('chain', 'domain', 'auto', 'none'):
        raise ValueError('Unknown tertiary mode: ' + tertiary_mode)
    path = Path(cif_file)
    if not path.is_file() or not path.stat().st_size:
        raise ValueError('Missing or empty CIF')
    sequences, ids = extract_sequences(load_structure(path), model_index)
    data = MMCIF2Dict(str(path))
    secondary = secondary_blocks(data)
    domains = (domain_ranges if domain_ranges is not None else
               ranges_from_category(data, 'pdbx_domain_range', forced_type='domain'))
    result = {}
    for chain, sequence in sequences.items():
        if chains is not None and chain not in chains:
            continue
        units = sorted(set(span for row in domains if row['chain_id'] == chain
                           for span in [_positions(row, ids[chain])] if span is not None))
        if tertiary_mode == 'domain' and not units:
            raise ValueError('No domain-range annotation for chain ' + chain)
        source = 'none'
        if tertiary_mode == 'chain' or (tertiary_mode == 'auto' and not units):
            units, source = [(0, len(sequence))], 'chain'
        elif tertiary_mode in ('domain', 'auto'):
            source = 'domain_csv' if domain_ranges is not None else 'pdbx_domain_range'
        else:
            units = []
        if any(right[0] < left[1] for left, right in zip(units, units[1:])):
            raise ValueError('Overlapping domain ranges cannot be encoded as disjoint scopes')
        result[chain] = dict(primary=sequence, residue_ids=ids[chain],
                             secondary=_secondary_units(secondary[chain], ids[chain]),
                             tertiary=[dict(start=start, end=end) for start, end in units],
                             tertiary_source=source)
    if not result:
        raise ValueError('No amino-acid chain recovered from structure')
    return result


def process(hierarchy):
    tokens = []
    for chain in hierarchy.values():
        sequence = chain['primary']
        scopes = chain['tertiary']
        # Clip secondary units at domain boundaries, avoiding crossed markers.
        cuts = {point for scope in scopes for point in (scope['start'], scope['end'])}
        secondary = []
        for unit in chain['secondary']:
            points = [unit['start']] + sorted(point for point in cuts
                      if unit['start'] < point < unit['end']) + [unit['end']]
            secondary.extend(dict(start=a, end=b, type=unit['type']) for a, b in zip(points, points[1:]))
        t_start = {scope['start'] for scope in scopes}
        t_end = {scope['end'] for scope in scopes}
        s_start = {unit['start']: unit['type'] for unit in secondary}
        s_end = {unit['end'] for unit in secondary}
        for i in range(len(sequence) + 1):
            if i in s_end:
                tokens.append('[secondary_end]')
            if i in t_end:
                tokens.append('[tertiary_end]')
            if i in t_start:
                tokens.append('[tertiary_start]')
            if i in s_start:
                tokens.extend(('[secondary_start]', '[' + s_start[i] + ']'))
            if i < len(sequence):
                tokens.append(sequence[i])
    return ' '.join(tokens)
