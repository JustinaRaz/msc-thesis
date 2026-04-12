from pathlib import Path
from data_analysis.src.morphology import get_morphological_annotations

pdf = Path("data_analysis/data/pdfs/Mokomes_skaityti_lietuviskai_suaugusiems.pdf")

def main():

    get_morphological_annotations(pdf_path = pdf)

if __name__ == "__main__":
    main()