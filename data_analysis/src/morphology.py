import csv
import re
from pathlib import Path

import fitz
import stanza
from data_analysis.src.logger import logger


def get_stanza():
    """
    Initiates Stanza for Lithuanian text processing.
    """
    logger.info("Downloading Lithuanian Stanza model.")
    stanza.download("lt")

    logger.info("Initializing Stanza processor.")
    processor = stanza.Pipeline(
        lang="lt", processors="tokenize,pos,lemma", tokenize_pretokenized=False
    )

    return processor


def get_pages(pdf: Path):
    """
    Gets the pages of the PDF.
    """
    pages = fitz.open(pdf)
    return pages


def clean_text(text):
    """
    Cleans the text by removing the unwanted symbols from the text that appear after reading-in the file.
    """
    text = text.replace("\n", "")
    text = text.replace("\xa0", " ")
    text = re.sub(r"(?<=\w)-(?=\w)", "", text)

    return text


def extract_text(
    pages, page_start: int, page_end: int, start_marker: str, end_marker: str
):
    """
    Extracts the text only of the stories, excluding the text in the corners of the page.

    Inputs:
        page_start (int): the page from which the extraction should start
        page_end (int): the page at which the extraction should end
        start_marker (str): a word / words from which the text should be kept
        end_marker (str): a word / words to which the text should be kept
    """

    text = ""

    for page_num in range(page_start, page_end):
        page = pages[page_num]
        blocks = page.get_text("blocks")

        for block in blocks:
            x0, y0, x1, y1, block_text, *_ = block

            if y0 < 40:
                continue
            if y0 > 620:
                continue

            text += block_text

    start = text.find(start_marker)
    end = text.find(end_marker)

    text = text[start:end].strip()
    cleaned_text = clean_text(text)

    return cleaned_text


def save_pos(doc, file_name: str):

    save_dir = f"data_analysis/data/morphological_annotations/pedagogic_lithuanian_corpus/c1/{file_name}.pos"

    with open(save_dir, "w", encoding="utf-8") as f:
        for sent in doc.sentences:
            for word in sent.words:
                form = word.text
                lemma = word.lemma
                upos = word.upos
                xpos = word.xpos if word.xpos else "_"
                feats = word.feats if word.feats else "_"

                f.write(f"{form}\t{lemma}\t{upos}\t{xpos}\t{feats}\n")
            f.write("\n")

    logger.info(f"File saved at {save_dir}.")


def get_morphological_annotations(pdf_path: Path):

    logger.info(f"Extracting the text from the PDF {pdf_path}.")
    pages = get_pages(pdf_path)

    logger.info("Extracting texts based on pre-specified pages.")
    text1 = extract_text(pages, 232, 245, "Tas simboliškas", "Klausimai ir užduotys")
    text2 = extract_text(pages, 536, 554, "Devyni kieti", "Klausimai ir užduotys")
    text3 = extract_text(pages, 575, 595, "Bepigu*", "Severiutė tauškė plepėjo")

    processor = get_stanza()

    logger.info("Performing morphological annotation. This might take long.")

    logger.info("Processing text 1/3.")
    doc_text1 = processor(text1)

    logger.info("Processing text 2/3.")
    doc_text2 = processor(text2)

    logger.info("Processing text 3/3.")
    doc_text3 = processor(text3)

    save_pos(doc_text1, "c1-1")
    save_pos(doc_text2, "c1-2")
    save_pos(doc_text3, "c1-3")


def get_morphological_diversity(data_dir: str):

    sentence_complexities = []

    data_dir = Path(data_dir)

    for file in data_dir.glob("*.pos"):
        with open(file, "r", encoding="utf-8") as f:
            sentence_feats = []

            for line in f:
                line = line.strip()

                # blank line = sentence boundary
                if line == "":
                    if sentence_feats:
                        unique_feats = set(sentence_feats)
                        sentence_complexities.append(len(unique_feats))
                        sentence_feats = []
                    continue

                parts = line.split("\t")

                # skip malformed rows
                if len(parts) < 5:
                    continue

                token, lemma, upos, morph_tag, features = parts[:5]

                if features != "_":
                    sentence_feats.extend(features.split("|"))

            # handle last sentence
            if sentence_feats:
                unique_feats = set(sentence_feats)
                sentence_complexities.append(len(unique_feats))

    avg_morph_complexity = sum(sentence_complexities) / len(sentence_complexities)

    return avg_morph_complexity


def get_morphological_overview():

    a1 = get_morphological_diversity("data_analysis/pedagogic_lithuanian_corpus/a1")
    b1 = get_morphological_diversity("data_analysis/pedagogic_lithuanian_corpus/b1")
    c1 = get_morphological_diversity("data_analysis/pedagogic_lithuanian_corpus/c1")

    rows = [
        ["CEFR_level", "avg_morphological_diversity"],
        ["A1", a1],
        ["B1", b1],
        ["C1", c1],
    ]

    save_dir = "data_analysis/data/REF_mean_morphological_diversity.csv"

    with open(save_dir, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
        logger.info(
            f"The average morhological diversity scores for reference texts are saved at {save_dir}."
        )
