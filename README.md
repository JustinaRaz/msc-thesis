# Master's Thesis
*School of Communication and Culture, Aarhus University*

Title of the thesis: **Interactive Tutoring for Learning Lithuanian as L2: CEFR-based Prompting Approach**

## 1. Overview
Repository contains the code for data simulation and analysis conducted as part of a master’s thesis project.

The repository is organized into two main components: `data_simulation/` and `data_analysis/`.

```
msc-thesis/
├── data_analysis/
│   ├── data/
|   |   ├── input/
|   |   └── output/
│   ├── plots/
│   ├── scripts/
│   ├── src/
│   └── analysis.py
│
├── data_simulation/
│   ├── configs/
|   |   ├── prompts/
|   |   |   ├── refinement.yaml
|   |   |   ├── student.yaml
|   |   |   └── tutor.yaml
|   |   └── models.toml
|   |
│   ├── output/
|   |   ├── refinement/
|   |   └── reproducibility/
|   |
│   ├── src/
|   |   ├── data_cleaning.py
|   |   ├── hf_gemma.py
|   |   ├── model_load.py
|   |   ├── models.py
|   |   ├── ref_evaluation.py
|   |   └── ref_thresholds.py
|   |
│   ├── tokens/
|   |   └── hf_token.txt              # Your token.
│   ├── simulate.py
│   └── simulate_self_refinement.py
│
├── justfile 
├── pyproject.toml 
├── settings.py
├── setup.sh 
└── README.md 
```

### 1.1 Data Simulation
As thesis is a partial replication, data simulation was performed using a similar pipeline as of the [previous study](https://github.com/INTERACT-LLM/Interact-LLM/tree/v1.0.3-alignment-drift/src/scripts/alignment_drift#%EF%B8%8F-overview).

#### 1.1.1 `data_simulation/` directories

| Directory | Purpose |
|-----------|---------|
| `configs/` | Configuration files defining LLMs and prompts. |
| `output/` | Generated datasets: tutor-student conversations. |
| `src/` | Source code implementing the simulation framework and supporting utilities. |
| `tokens/` | HuggingFace authentication token. |

#### 1.1.2 Scripts

| Script | Purpose |
|---------|---------|
| `simulate.py` | Runs the standard data simulation workflow (replication). |
| `simulate_self_refinement.py` | Runs simulations incorporating the self-refinement procedure. |

### 1.2 Data Analysis

Component includes the analysis of the simulated conversational data, including pre-processing, statistical analyses, visualizations.

#### 1.2.1 `data_analysis/` directories

| Directory | Purpose |
|-----------|---------|
| `data/` | Data files. |
| `plots/` | Plots displayed in the paper of the project. |
| `scripts/` | Scripts to be run using specifications in justfile. |
| `src/` | Source code. |

## 2. Reproducibility

To reproduce the code, clone the repository and set the working directory to the project root:
```python
cd msc-thesis
```
Install required dependencies by running:
```bash
bash setup.sh
```

To see all available `just` recipies:
```bash
just --list
```

### 2.1 Data Simulation
Simulation of tutor-student conversations is extremely resource- and time-intensive. Nevertheless, all conversations can be simulated by running the following `just` recipes:

```bash
just reproduce
just refine
```

### 2.2 Data Analysis
Justfile contains all recipies for data analysis. 

**Note**: the following directories, namely:
- data_analysis/data/output/reference_data/morph_annotations/a1 
- data_analysis/data/output/reference_data/morph_annotations/b1 

do not contain reference files, as these are not publically available. In need, please contact the authors who were involved in curating [**Lithuanian Pedagogic Corpus**](https://www.vdu.lt/cris/entities/product/e6732361-3893-44be-8b1e-81d2773511ce).