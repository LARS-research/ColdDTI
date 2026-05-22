# Dataset information

## Overview

This project uses four benchmark datasets for drug-target interaction (DTI) prediction:

- **BindingDB**
- **BioSNAP**
- **Human**
- **DrugBank**

All datasets were obtained from their official sources or official benchmark release pages and were further processed into a unified format for model training and evaluation.

## Data Sources

### 1. BindingDB
BindingDB is a public database of experimentally measured binding affinities between proteins and small molecules.

- Official website: https://www.bindingdb.org/
- Official download page: https://www.bindingdb.org/rwd/bind/chemsearch/marvin/Download.jsp

### 2. BioSNAP
BioSNAP provides curated biomedical network datasets. In this project, we use the drug-target interaction benchmark released through the Stanford SNAP BioSNAP collection.

- Official BioSNAP dataset page: https://snap.stanford.edu/biodata/datasets/10015/10015-ChG-TargetDecagon.html
- BioSNAP collection page: https://snap.stanford.edu/biodata/

### 3. Human
The Human dataset used in this project is a widely adopted DTI benchmark dataset from the benchmark release used in prior DTI studies.

- Official benchmark release source: https://github.com/lifanchen-simm/transformerCPI
- Human dataset file: https://github.com/lifanchen-simm/transformerCPI/blob/master/Human%2CC.elegans/dataset/human_data.txt

### 4. DrugBank
DrugBank is a comprehensive knowledgebase integrating drug information with drug-target associations.

- Official website: https://go.drugbank.com/
- Official release/download page: https://go.drugbank.com/releases/latest

## Download Policy

The raw datasets were obtained from the official websites or official benchmark release pages listed above.


## Directory Structure

```bash
Dataset/
├── bindingdb/
├── biosnap/
├── human/
├── drugbank/
└── README.md