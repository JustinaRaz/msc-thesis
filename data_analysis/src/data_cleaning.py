import polars as pl
import regex as re
from lingua import Language, LanguageDetectorBuilder
from data_analysis.utils.logger import logger
from data_analysis.utils.models.metadata import DataFile
from pathlib import Path


class DataCleaner:
    def __init__(self, english_threshold: float = 0.5):

        self.english_threshold = english_threshold

        self.detector = LanguageDetectorBuilder.from_languages(
            Language.ENGLISH, Language.LITHUANIAN
        ).build()

        # aggregated statistics for all files
        self.file_stats = []

        # counters for the currently processed file
        self.reset_counters()

    def reset_counters(self):

        self.repetitive_words = 0
        self.invalid_sentences = 0
        self.long_words = 0
        self.invalid_char_words = 0

    def set_metadata(self, file: DataFile):

        self.current_file = file
        self.reset_counters()

    def finalize_file(self):

        self.file_stats.append({
            "model": self.current_file.model,
            "prompt_type": self.current_file.type,
            "language": self.current_file.language,
            "cefr": self.current_file.cefr,
            "file": self.current_file.file_name,
            "repetitive_words_removed": self.repetitive_words,
            "invalid_sentences_removed": self.invalid_sentences,
            "long_words_removed": self.long_words,
            "invalid_char_words_removed": self.invalid_char_words,
        })

    @staticmethod
    def split_to_sentences(text: str, keep_punctuation = True):

        if keep_punctuation:
            sentences = re.split(r"([.?!])", text)
            result = []

            for i in range(0, len(sentences), 2):
                sentence = sentences[i].strip()
                if i + 1 < len(sentences):
                    sentence += sentences[i + 1]
                if sentence:
                    result.append(sentence)

            return result

        sentences = re.split(r"[.!?]+", text)
        return sentences

    @staticmethod
    def strip_boundary_chars(sentence: str) -> str:
        return sentence.strip(" :()")

    @staticmethod
    def tokenize(sentence):
        """
        Lowercase, remove punctuation, return word tokens.
        """
        tokenized_sentence = re.findall(r"\w+", sentence.lower())
        return tokenized_sentence

    @staticmethod
    def remove_non_text(text: str) -> str:
        """
        Removes unnecessary characters (emojis, new line symbols) from the sentence.
        """
        return re.sub(r"[^\p{L}\p{N}\s\.,?!:;'\-\(\)\"]+", "", text)

    def contains_english(self, sentence: str) -> bool:

        ENGLISH_WORD_PATTERN = re.compile(
            r"\b(the|and|is|are|a|an|of|to|in|it|you|I|this|that|with|for)\b",
            re.IGNORECASE,
        )

        if ENGLISH_WORD_PATTERN.search(sentence):
            return True

        confidence_values = self.detector.compute_language_confidence_values(sentence)

        for confidence in confidence_values:
            if (
                confidence.language == Language.ENGLISH
                and confidence.value >= self.english_threshold
            ):
                return True

        return False
    
    def clean_words(self, sentence: str) -> str:

        allowed_pattern = re.compile(r"^[a-zA-ZąčęėįšųūžĄČĘĖĮŠŲŪŽ]+$")

        # keep punctuation as separate tokens
        tokens = re.findall(r"\w+|[.,?!:;]", sentence)

        cleaned_tokens = []

        prev_word = None

        for token in tokens:

            # keep punctuation without processing
            if re.match(r"[.,?!:;]", token):
                cleaned_tokens.append(token)
                continue

            word = token

            if len(word) > 40:
                self.long_words += 1
                continue

            if prev_word and word.lower() == prev_word.lower():
                self.repetitive_words += 1
                continue

            if not allowed_pattern.match(word):
                self.invalid_char_words += 1
                continue

            cleaned_tokens.append(word)
            prev_word = word

        # rebuild sentence
        sentence = " ".join(cleaned_tokens)

        # remove space before punctuation
        sentence = re.sub(r"\s+([.,?!:;])", r"\1", sentence)

        return sentence


    def clean_text(self, text: str) -> str:

        sentences = self.split_to_sentences(text)

        cleaned_sentences = []

        for sentence in sentences:

            if self.contains_english(sentence):
                self.invalid_sentences += 1
                continue

            sentence = self.strip_boundary_chars(sentence)

            cleaned_sentence = self.remove_non_text(sentence)

            cleaned_sentence = self.clean_words(cleaned_sentence)

            if cleaned_sentence.strip():
                cleaned_sentences.append(cleaned_sentence.strip())

        return " ".join(cleaned_sentences)

    def export_report(self, output_dir: Path):

        df = pl.DataFrame(self.file_stats)

        path = output_dir / "cleaning_report.csv"

        df.write_csv(path)

        return path