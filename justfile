# DATA SIMULATION

reproduce:
    #!/usr/bin/env bash
    PROMPT_LEVELS=("A1" "B1" "C1")
    PROMPT_LANGUAGES=("Lithuanian" "English")
    MODEL_SIZES=("small" "medium" "large")
    STUDENT_CONSTRAIN_OPTIONS=("" "--student_constrain")

    for level in "${PROMPT_LEVELS[@]}"; do
        for language in "${PROMPT_LANGUAGES[@]}"; do
            for model_size in "${MODEL_SIZES[@]}"; do
                for constrain_flag in "${STUDENT_CONSTRAIN_OPTIONS[@]}"; do
                    .venv/bin/python data_simulation/simulate.py \
                    --prompt_cefr_level "$level" \
                    --prompt_language "$language" \
                    --model_size "$model_size" \
                    $constrain_flag
                done
            done
        done
    done

refine:
    #!/usr/bin/env bash
    PROMPT_LEVELS=("B1" "C1")
    PROMPT_LANGUAGES=("Lithuanian")
    MODEL_SIZES=("medium")

    for level in "${PROMPT_LEVELS[@]}"; do
        for language in "${PROMPT_LANGUAGES[@]}"; do
            for model_size in "${MODEL_SIZES[@]}"; do
                .venv/bin/python data_simulation/simulate_self_refinement.py \
                    --prompt_cefr_level "$level" \
                    --prompt_language "$language" \
                    --model_size "$model_size"
            done
        done
    done


# ---------------- REFERENCE TEXTS
ref_morph_annot:
    .venv/bin/python -m data_analysis.scripts.reference.reference_morphological_tagging

ref_surface:
    .venv/bin/python -m data_analysis.scripts.reference.reference_surface_level_stats

ref_sanity_check:
    .venv/bin/python -m data_analysis.scripts.reference.reference_morphology

ref_mdd:
    .venv/bin/python -m data_analysis.scripts.reference.reference_mdd

# ---------------- DATA PREPARATION

get_data:
    .venv/bin/python -m data_analysis.scripts.llm.get_clean_data

# ---------------- LLM METRICS/MORPHOLOGY

llm_metrics:
    .venv/bin/python -m data_analysis.scripts.llm.metrics

llm_annotate:
    .venv/bin/python -m data_analysis.scripts.llm.llm_annotations

llm_morph_summaries:
    .venv/bin/python -m data_analysis.scripts.llm.linguistic_overview


# ---------------- PRE-PROCESSING

cleaning_report_plot:
    .venv/bin/python -m data_analysis.scripts.preprocessing.plot_cleaning_report

error_rates_plots:
    .venv/bin/python -m data_analysis.scripts.preprocessing.plot_error_rates


# ---------------- ALIGNMENT DRIFT VISUALIZATION PLOTTING 

averages:
    .venv/bin/python -m data_analysis.scripts.alignment_drift.averages

densities:
    .venv/bin/python -m data_analysis.scripts.alignment_drift.densities

# ---------------- SELF-REFINEMENT

refinement_metrics:
    .venv/bin/python -m data_analysis.scripts.refinement.refinement_metrics

plot_refinement:
    .venv/bin/python -m data_analysis.scripts.refinement.plot_refinement


# ---------------- LLM VERSUS REFERENCE

llm_ref_summary:
    .venv/bin/python -m data_analysis.scripts.llm_ref_compare.morph_freqs_combine

plot_verbs:
    .venv/bin/python -m data_analysis.scripts.llm_ref_compare.plot_verbs

plot_nouns:
    .venv/bin/python -m data_analysis.scripts.llm_ref_compare.plot_nouns

# ---------------- LINEAR MIXED-EFFECTS

plot_tables:
    .venv/bin/python -m data_analysis.scripts.mixed_effects.plot_tables

# ---------------- DISTANCE/MUTUAL ADAPTABILITY PLOTS

distance:
    .venv/bin/python -m data_analysis.scripts.alignment_drift.distance