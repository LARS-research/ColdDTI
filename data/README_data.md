# Current ColdDTI data

The four paper datasets use split seed=57. See the
[reproduction guide](../README.md) for protocols, scope modes and runnable commands.

Each dataset directory contains:

- `observations.csv`: cleaned source benchmark pairs before structure filtering.
- `source_proteins.csv`, `smiles.csv`: standalone entity input snapshots.
- `protein.csv`: the currently selected structure-tagged proteins; row numbers define `protein_nid`.
- `structure_coverage.csv`: accepted/rejected structures, reasons and scope counts.
- `preprocessing.json`: actual tag modes, source cleaning and protocol definitions.
- `split_statistics.csv`: each partition's samples/entities/positive/negative counts.
- `feature_status.json`: versioned branch manifests and file/input digests; only a fully validated `ready=true` cache can be loaded.
- `cluster_groups/`: fixed upstream cluster members, merged by actual entity identity.

Every split uses the same protein index table. Basic cold settings hold out the
corresponding entity identities; cluster settings hold out both identity groups.
DrugBank entity-date temporal splitting uses a 2010 cutoff and pair-disjoint
historical training/validation versus future-entity test examples.

CIFs, frozen encoder checkpoints and generated `.npy` feature arrays are excluded
from Git. Generate features locally with `get_emb.py` using the supplied tagged
protein table and local encoders; CIFs are needed only to rebuild structure tags.
The cache status prevents stale features from being used for training. Data
statistics after all filtering are in the root `split_statistics.csv`.

The BindingDB benchmark source is `bindingDB_mapped.csv`; the reported filtering
statistics cover processing of this benchmark input snapshot.
`bindingdb/mapping_statistics.json` records the recovered labeled-benchmark
mapping and subsequent cleaning/structure-filtering counts.

DrugBank also includes `drug_identifiers.csv`, joined by actual SMILES, and
`protein_regions.csv`, containing transmembrane annotations for the case-study
muscarinic receptor proteins. Region positions use one-based source-sequence
numbering; sequence SHA256 values identify their coordinate reference.
