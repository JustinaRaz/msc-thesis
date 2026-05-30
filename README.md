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

### `data_simulation/`

#### 1.1.1 Directories

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

### `data_analysis/`

#### 1.2.1 Directories

| Directory | Purpose |
|-----------|---------|
| `data/` | Configuration files defining LLMs and prompts. |
| `plots/` | Generated datasets: tutor-student conversations. |
| `scripts/` | Source code implementing the simulation framework and supporting utilities. |
| `src/` | HuggingFace authentication token. |

#### 1.2.2 Scripts

| Script | Purpose |
|---------|---------|
| `simulate.py` | Runs the standard data simulation workflow (replication). |
| `simulate_self_refinement.py` | Runs simulations incorporating the self-refinement procedure. |

## To Reproduce the Study

In order to reproduce the code, clone the repository and set the working directory to the project root:
```python
cd msc-thesis
```
Then, to reproduce the code, install the following dependencies by running:
```python
sudo apt update
sudo apt install -y libhunspell-dev hunspell
sudo apt install hunspell-lt
uv sync
sudo apt install just
```