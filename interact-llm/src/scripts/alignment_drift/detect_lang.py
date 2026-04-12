import re

from lingua import Language, LanguageDetectorBuilder

LANGUAGES = [Language.ENGLISH, Language.SPANISH, Language.CHINESE]

test_text_with_english = "Me alegra saber que estás disfrutando de la clase. A mí también me parece divertida hoy, especialmente porque vamos a hablar sobre las festividades en España. ¿Sabías que la fiesta más famosa es el Carnaval? (I'm glad you're enjoying the class. I find it fun today too, especially because we're going to talk about festivals in Spain. Did you know that the most famous party is Carnival?)"

test_text_without_english = "Me alegra saber que estás disfrutando de la clase. A mí también me parece divertida hoy, especialmente porque vamos a hablar sobre las festividades en España. ¿Sabías que la fiesta más famosa es el Carnaval?"

test_text_CHINESE = "¡Genial! Has hecho un excelente trabajo改进你的翻译和修订。以下是稍作调整后更流畅和完善的一些文字：### 结构化的短篇故事《公园的一天》> **Un Día en el Parque** Antes de unos días, decidimos ir al parque con mis amigos María y Juan. Ese día estaba soleado y hermoso, perfecto para pasar una jornada divertida al aire libre. Primero, caminamos por las diferentes atracciones del parque, disfrutando de los jardines y observando a las aves que revoloteaban por todos lados. Luego, nos dirigimos a la zona de juegos, donde pasamos gran parte del tiempo. Fue especialmente divertido el día que jugamos al volante virtual, donde pudimos vivir la experiencia de conducir sin peligro. Finalmente, cenamos en uno de los restaurantes cercanos, degustando exquisita comida. Ese día no solo disfrutamos de la diversión, sino que también aprendimos sobre la importancia de cuidar nuestros cuerpos al realizar ejercicios en el parque. ### 在线游戏以提高发音 我会继续尝试这些游戏：- **Duolingo中的音素拼图**：从基础开始，反复练习直到熟悉每个单词的发音。记得多次听并模仿发音"

def _split_text(text: str) -> list[str]:
    sents = re.split(r"([.?!])", text)  # capture sent delimiters
    result = []

    for i in range(0, len(sents), 2):  # process in pairs
        sentence = sents[i].strip()
        if i + 1 < len(sents):  # if there is a delimiter, attach it
            sentence += sents[i + 1]
        if sentence:
            result.append(sentence)

    return result


def _detect_lang(
    text: list[str] | str,
    languages_to_consider: list[Language] = LANGUAGES,
    language_thresholds: list[tuple[Language, float]] = [(Language.ENGLISH, 0.80), (Language.CHINESE, 0.80)],
) -> bool:
    """
    Returns true if any language threshold (specific language, confidence that X contains language) is met
    
    Args:
        text: Text to detect language in
        languages_to_consider: Languages to consider in detection
        language_thresholds: List of (language, confidence) thresholds to check
    
    Returns:
        bool: True if any language threshold is met, False otherwise
    """
    if not isinstance(text, list):
        sents = _split_text(text)
    else:
        sents = text

    detector = LanguageDetectorBuilder.from_languages(*languages_to_consider).build()

    for sent in sents:
        confidence_values = detector.compute_language_confidence_values(sent)

        for threshold in language_thresholds:
            for confidence in confidence_values:
                if (
                    confidence.language == threshold[0]
                    and confidence.value >= threshold[1]
                ):
                    print(
                        f"[INFO]: Text contains at least one sentence with {threshold[0].name} (confidence of {threshold[1]})"
                    )
                    return True  # If any sentence meets any threshold, return True

    return False  # If no sentences meet any threshold, return False

if __name__ == "__main__":
    # Case 1: Text with both Spanish and English
    print("Test 1: Detection with English")
    detection_with_english = _detect_lang(text=test_text_with_english)
    print(detection_with_english, "\n")

    # Case 2: Text fully in Spanish (no English present)
    print("Test 2: Detection without English")
    detection_without_english = _detect_lang(text=test_text_without_english)
    print(detection_without_english, "\n")

    print("Test 2: Detection Chinese/Spanish MIX")
    detection_with_chinese = _detect_lang(text=test_text_CHINESE)
    print(detection_with_chinese, "\n")
