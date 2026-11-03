"""Prepare residue-aligned structure tags and all dataset split indices."""
from preprocess.structure import (load_structure, extract_sequences, ss_type_to_char,
                                 extract_secondary_structure_blocks, extract_hierarchical, process)

if __name__ == '__main__':
    from preprocess.prepare import main
    main()
