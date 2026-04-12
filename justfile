reqs:
    sudo apt install -y libhunspell-dev hunspell
    sudo apt install hunspell-lt
    #wget https://apertium.projectjj.com/apt/install-nightly.sh -O - | sudo bash
    #sudo apt-get install cg3 hfst

# To clean and combine LLM-simulated text into one file:
get_data:
    .venv/bin/python -m data_analysis.scripts.get_clean_data

# To extract the Text descriptives for the LLM-simulated clean text:
get_metrics:
    .venv/bin/python -m data_analysis.scripts.get_metrics

# To plot the descriptive plots:
get_descriptive_plots:
    .venv/bin/python -m data_analysis.scripts.get_descriptive_plots

get_error_rate_plots:
    reqs
    .venv/bin/python -m data_analysis.scripts.get_error_rate_plots

get_cleaning_report_plots:
    .venv/bin/python -m data_analysis.scripts.get_cleaning_report_plots

morph_annot:
    .venv/bin/python -m data_analysis.scripts.extract_morphology

mixed_effects:
    .venv/bin/python -m data_analysis.scripts.mixed_effects

llm_annotate:
    .venv/bin/python -m data_analysis.scripts.llm_annotations

ref_sanity_check:
    .venv/bin/python -m data_analysis.scripts.morphology.reference_morphology