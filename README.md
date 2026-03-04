# MSc Cognitive Science
## Aarhus University


To run the code, clone the repository and set the working directory to the project root:
```python
git msc-thesis
```

Then, create the virtual environment by running:
```python
uv sync
```

In case of the hunspell-related error, run the following. Then try to create the virtual environment again.
```python
sudo apt update
sudo apt install -y libhunspell-dev hunspell
```

Different parts of the code can be run using justfile defined commands:
```python
just get_data # To merge and clean LLM-simulated dialogues.
just get_metrics # To get metrics
```
