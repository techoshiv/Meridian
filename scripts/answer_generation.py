"""Generate evidence-grounded answers with a locally running Ollama model."""

from __future__ import annotations

import json
import re
import socket
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2:latest"
DEFAULT_TIMEOUT_SECONDS = 120.0
MAX_GENERATED_ANSWER_WORDS = 60
ABSTENTION = "I don't have sufficient evidence in the retrieved sources to answer that."
ROMAN_HINDI_MARKERS = {
    "hai", "hain", "ka", "ke", "ki", "kya", "kitna", "liye", "mein", "nahi",
    "par", "se", "tha", "thi", "wala", "wali", "wale",
}
GENERIC_SECTION_TERMS = {
    "amount", "charges", "conditions", "eligibility", "interest", "policy",
    "rates", "rule", "rules", "section", "english", "hindi",
}
DOCUMENT_TERMS = {
    "bank", "digital", "fd", "grih", "home", "locker", "loan", "personal",
    "rin", "savings",
}
PRODUCT_TERMS = DOCUMENT_TERMS - {"bank", "loan"}
NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
    "hundred": 100,
    "एक": 1,
    "दो": 2,
    "तीन": 3,
    "चार": 4,
    "पाँच": 5,
    "पांच": 5,
    "छह": 6,
    "सात": 7,
    "आठ": 8,
    "नौ": 9,
    "दस": 10,
    "ग्यारह": 11,
    "बारह": 12,
    "तेरह": 13,
    "चौदह": 14,
    "पंद्रह": 15,
    "पन्द्रह": 15,
    "सोलह": 16,
    "सत्रह": 17,
    "अठारह": 18,
    "उन्नीस": 19,
    "बीस": 20,
    "तीस": 30,
    "चालीस": 40,
    "पचास": 50,
    "साठ": 60,
    "सत्तर": 70,
    "अस्सी": 80,
    "नब्बे": 90,
    "सौ": 100,
}
DEVANAGARI_INDEPENDENT_VOWELS = {
    "अ": "a",
    "आ": "aa",
    "इ": "i",
    "ई": "ii",
    "उ": "u",
    "ऊ": "uu",
    "ऋ": "ri",
    "ए": "e",
    "ऐ": "ai",
    "ओ": "o",
    "औ": "au",
}
DEVANAGARI_VOWEL_SIGNS = {
    "ा": "aa",
    "ि": "i",
    "ी": "ii",
    "ु": "u",
    "ू": "uu",
    "ृ": "ri",
    "े": "e",
    "ै": "ai",
    "ो": "o",
    "ौ": "au",
}
DEVANAGARI_CONSONANTS = {
    "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "ng",
    "च": "ch", "छ": "chh", "ज": "j", "झ": "jh", "ञ": "ny",
    "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh", "ण": "n",
    "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n",
    "प": "p", "फ": "ph", "ब": "b", "भ": "bh", "म": "m",
    "य": "y", "र": "r", "ल": "l", "व": "v", "श": "sh",
    "ष": "sh", "स": "s", "ह": "h", "क़": "q", "ख़": "kh",
    "ग़": "gh", "ज़": "z", "ड़": "r", "ढ़": "rh", "फ़": "f",
}
DEVANAGARI_DIACRITICS = {"ं": "n", "ँ": "n", "ः": "h"}
DEVANAGARI_SIGNS = {"।": ".", "॥": "."}
ANSWER_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in",
    "is", "it", "of", "on", "or", "that", "the", "this", "to", "was", "were",
    "के", "की", "का", "को", "से", "में", "पर", "है", "हैं", "हो", "और",
    "एक", "यह", "इस", "किया", "जाता", "जाएगा", "जाएगी", "करने", "नहीं",
}
QUANTITATIVE_QUESTION_TERMS = {
    "amount", "annual", "charge", "charges", "cost", "days", "duration", "fee",
    "fees", "how", "limit", "maximum", "max", "minimum", "min", "month",
    "months", "much", "many", "penalty", "percent", "percentage", "rate",
    "rates", "tenure", "time", "year", "years",
    "अधिकतम", "अवधि", "कितना", "कितनी", "कितने", "दर", "दंड", "दिन", "न्यूनतम",
    "प्रतिशत", "महीने", "मास", "सीमा", "शुल्क", "वर्ष",
}


class OllamaRequestError(RuntimeError):
    """Raised when Ollama returns an unsuccessful HTTP response."""


class OllamaConnectionError(ConnectionError):
    """Raised when the local Ollama service cannot be reached."""


class OllamaTimeoutError(TimeoutError):
    """Raised when Ollama does not respond within the configured timeout."""


def _source_text(result: dict[str, Any]) -> str:
    text = result.get("source_text")
    if not isinstance(text, str) or not text.strip():
        text = result.get("snippet")
    if not isinstance(text, str):
        text = result.get("text")
    return text.strip() if isinstance(text, str) else ""


def _citations(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    citations: list[dict[str, Any]] = []
    seen: set[tuple[str, tuple[int, ...], str]] = set()
    for result in results:
        source_pdf = result.get("source_pdf")
        pages = result.get("pages")
        chunk_id = result.get("chunk_id")
        if not isinstance(source_pdf, str) or not source_pdf.strip():
            continue
        if not isinstance(pages, list) or not all(
            isinstance(page, int) and not isinstance(page, bool) for page in pages
        ):
            continue
        normalized_pages = tuple(sorted(set(pages)))
        if not normalized_pages:
            continue
        key = (source_pdf, normalized_pages, chunk_id if isinstance(chunk_id, str) else "")
        if key in seen:
            continue
        seen.add(key)
        citation: dict[str, Any] = {
            "source_pdf": source_pdf,
            "pages": list(normalized_pages),
        }
        if isinstance(chunk_id, str) and chunk_id:
            citation["chunk_id"] = chunk_id
        section_ids = result.get("section_ids")
        if isinstance(section_ids, list) and all(isinstance(section, str) for section in section_ids):
            citation["section_ids"] = section_ids
        citations.append(citation)
    return citations


def _answer_results(results: list[dict[str, Any]], question: str) -> list[dict[str, Any]]:
    ranked_results = [result for result in results if isinstance(result, dict)]
    if not ranked_results:
        return []
    primary = ranked_results[0]
    question_terms = set(re.findall(r"[a-z0-9]+", question.lower()))
    primary_scope = {
        token
        for section in primary.get("section_ids", [])
        if isinstance(section, str)
        for token in section.lower().split("-")
        if token and token not in GENERIC_SECTION_TERMS and token not in DOCUMENT_TERMS
    }
    primary_domains = {
        token
        for section in primary.get("section_ids", [])
        if isinstance(section, str)
        for token in section.lower().split("-")
        if token in PRODUCT_TERMS
    }
    selected = []
    for result in ranked_results:
        if result is primary:
            selected.append(result)
            continue
        result_scope = {
            token
            for section in result.get("section_ids", [])
            if isinstance(section, str)
            for token in section.lower().split("-")
            if token and token not in GENERIC_SECTION_TERMS and token not in DOCUMENT_TERMS
        }
        result_domains = {
            token
            for section in result.get("section_ids", [])
            if isinstance(section, str)
            for token in section.lower().split("-")
            if token in PRODUCT_TERMS
        }
        result_terms = {
            token
            for section in result.get("section_ids", [])
            if isinstance(section, str)
            for token in section.lower().split("-")
            if token and token not in DOCUMENT_TERMS
        }
        if (
            primary_domains & result_domains
            and (primary_scope & result_scope or result_terms & question_terms)
        ):
            selected.append(result)
    return selected


def _language_script_instruction(question: str) -> str:
    """Return explicit output-language guidance based on the question's script."""
    if re.search(r"[\u0900-\u097f]", question):
        return "Write the answer in Hindi using Devanagari script. Do not repeat these instructions."
    words = set(re.findall(r"[a-z]+", question.lower()))
    if len(words & ROMAN_HINDI_MARKERS) >= 2:
        return (
            "Write the answer in Romanised Hindi using the Latin alphabet, not Devanagari. "
            "Do not repeat these instructions."
        )
    return "Write the answer in English using the Latin alphabet. Do not repeat these instructions."


def _is_romanized_hindi_question(question: str) -> bool:
    if re.search(r"[\u0900-\u097f]", question):
        return False
    words = set(re.findall(r"[a-z]+", question.lower()))
    return len(words & ROMAN_HINDI_MARKERS) >= 2


def _romanize_devanagari(text: str) -> str:
    output: list[str] = []
    index = 0
    while index < len(text):
        character = text[index]
        if character in DEVANAGARI_INDEPENDENT_VOWELS:
            output.append(DEVANAGARI_INDEPENDENT_VOWELS[character])
        elif character in DEVANAGARI_CONSONANTS:
            output.append(DEVANAGARI_CONSONANTS[character])
            next_index = index + 1
            if next_index < len(text) and text[next_index] in DEVANAGARI_VOWEL_SIGNS:
                output.append(DEVANAGARI_VOWEL_SIGNS[text[next_index]])
                index = next_index
            elif next_index < len(text) and text[next_index] == "्":
                index = next_index
            elif next_index == len(text) - 1 or text[next_index].isspace() or text[next_index] in "।॥,.;:!?":
                pass
            else:
                output.append("a")
        elif character in DEVANAGARI_DIACRITICS:
            output.append(DEVANAGARI_DIACRITICS[character])
        elif character in DEVANAGARI_SIGNS:
            output.append(DEVANAGARI_SIGNS[character])
        elif character == "्":
            pass
        else:
            output.append(character)
        index += 1

    romanized = "".join(output)
    return re.sub(r"\s+", " ", romanized).strip()


def _numeric_values(text: str) -> set[int]:
    values: set[int] = set()
    for token in re.findall(r"[a-z]+|[\u0900-\u097f]+|[0-9]+|[०-९]+", text.lower()):
        if token.isdecimal():
            values.add(int(token.translate(str.maketrans("०१२३४५६७८९", "0123456789"))))
        elif token in NUMBER_WORDS:
            values.add(NUMBER_WORDS[token])
    return values


def _question_requests_numeric_fact(question: str) -> bool:
    terms = set(re.findall(r"[a-z]+|[\u0900-\u097f]+", question.casefold()))
    return bool(terms & QUANTITATIVE_QUESTION_TERMS)


def _has_source_term_support(answer: str, source_blocks: list[str]) -> bool:
    answer_uses_devanagari = bool(re.search(r"[\u0900-\u097f]", answer))
    source_uses_devanagari = any(
        re.search(r"[\u0900-\u097f]", text) for text in source_blocks
    )
    if answer_uses_devanagari != source_uses_devanagari:
        return True

    def terms(text: str) -> set[str]:
        return {
            token.strip("।॥,.;:()[]{}")
            for token in re.findall(r"[a-z]+|[\u0900-\u097f]+", text.lower())
            if token.strip("।॥,.;:()[]{}")
            and token.strip("।॥,.;:()[]{}") not in NUMBER_WORDS
            and token.strip("।॥,.;:()[]{}") not in ANSWER_STOPWORDS
        }

    answer_terms = terms(answer)
    source_terms = set().union(*(terms(text) for text in source_blocks))
    return len(answer_terms & source_terms) >= 2


def _format_source_fallback(source_text: str) -> str:
    """Remove a leading numbered provision heading and normalize source whitespace."""
    without_heading = re.sub(
        r"^\s*\d+[.)]\s*[^-\r\n]+?\s*-\s*",
        "",
        source_text,
        count=1,
    )
    if without_heading == source_text:
        without_heading = re.sub(
            r"^\s*\d+[.)]\s*[^-\r\n]+\r?\n",
            "",
            source_text,
            count=1,
        )
    return re.sub(r"\s+", " ", without_heading).strip()


def _relevant_table_fallback(question: str, source_blocks: list[str]) -> str | None:
    question_terms = {
        token
        for token in re.findall(r"[a-z]+|[\u0900-\u097f]+", question.casefold())
        if token not in ANSWER_STOPWORDS and token not in NUMBER_WORDS
    }
    question_numbers = _numeric_values(question)
    asks_for_greater_than = bool(
        question_terms & {"above", "exceed", "exceeds", "greater", "अधिक", "ऊपर"}
    )
    asks_for_at_most = bool(
        question_terms & {"upto", "तक", "below", "under", "less", "कम"}
    )
    candidates: list[tuple[int, str, str, str]] = []

    for source in source_blocks:
        lines = [line.strip() for line in source.splitlines() if "|" in line]
        rows = [line for line in lines if len(line.split("|")) >= 2]
        if len(rows) < 2:
            continue
        header = rows[0]
        for row in rows[1:]:
            row_terms = {
                token
                for token in re.findall(r"[a-z]+|[\u0900-\u097f]+", row.casefold())
                if token not in ANSWER_STOPWORDS and token not in NUMBER_WORDS
            }
            row_numbers = _numeric_values(row)
            score = len(question_terms & row_terms)
            if question_numbers:
                score += 3 * len(question_numbers & row_numbers)
            if asks_for_greater_than:
                if row_terms & {"above", "greater", "अधिक"}:
                    score += 3
                if row_terms & {"upto", "तक", "below", "under", "कम"}:
                    score -= 4
            if asks_for_at_most:
                if row_terms & {"upto", "तक", "below", "under", "कम"}:
                    score += 3
                if row_terms & {"above", "greater", "अधिक"}:
                    score -= 4
            if row_numbers and question_numbers and not question_numbers.intersection(row_numbers):
                continue
            columns = [column.strip() for column in row.split("|")]
            candidates.append((score, header, " | ".join(columns[1:]), " | ".join(columns[:-1])))

    if not candidates:
        return None
    candidates.sort(key=lambda candidate: candidate[0], reverse=True)
    best_score, header, value, condition = candidates[0]
    if best_score < 3 or (len(candidates) > 1 and best_score == candidates[1][0]):
        return None

    header_columns = [column.strip() for column in header.split("|")]
    value_columns = [column.strip() for column in value.split("|")]
    result_parts: list[str] = []
    if len(header_columns) == 2 and len(value_columns) == 1:
        result_parts.append(f"{header_columns[1]}: {value_columns[0]}")
    else:
        result_parts.append(value)
    condition = condition.strip(" |")
    if condition:
        result_parts.append(condition)

    date_matches = list(re.finditer(
        r"\b\d{1,2}\s+(?:जनवरी|फरवरी|मार्च|अप्रैल|मई|जून|जुलाई|अगस्त|सितंबर|अक्टूबर|नवंबर|दिसंबर|"
        r"January|February|March|April|May|June|July|August|September|October|November|December)"
        r"\s+\d{4}\b",
        " ".join(source_blocks),
        re.IGNORECASE,
    ))
    if date_matches:
        result_parts.append(f"({date_matches[-1].group(0)} से)")
    return " ".join(result_parts).strip() + "।"


def _fallback_for_question(source_text: str, question: str) -> str:
    formatted = _format_source_fallback(source_text)
    if _is_romanized_hindi_question(question):
        return _romanize_devanagari(formatted)
    return formatted


def _requested_card_types(question: str) -> set[str]:
    return {
        card_type
        for card_type in ("credit", "debit")
        if re.search(rf"\b{card_type}[\s-]+cards?\b", question, re.IGNORECASE)
    }


def _answer_repeats_question(answer: str, question: str) -> bool:
    normalize = lambda text: "".join(
        re.findall(r"[a-z0-9\u0900-\u097f]+", text.casefold())
    )
    return bool(normalize(question)) and normalize(answer) == normalize(question)


def answer_question(
    question: str,
    retrieval_result: dict[str, Any],
    *,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Answer from retrieved source excerpts, citing only their actual chunk metadata.

    Connection, timeout, and HTTP failures raise explicit exceptions. Insufficient
    retrieval evidence abstains without making an HTTP request.
    """
    results = retrieval_result.get("results")
    if (
        retrieval_result.get("status") == "no_sufficient_evidence"
        or not isinstance(results, list)
        or not results
    ):
        return {"answer": ABSTENTION, "citations": [], "found": False}

    answer_results = _answer_results(results, question)
    source_blocks = [
        text
        for result in answer_results
        if (text := _source_text(result))
    ]
    citations = _citations(answer_results)
    if not source_blocks or not citations:
        return {"answer": ABSTENTION, "citations": [], "found": False}
    requested_card_types = _requested_card_types(question)
    if requested_card_types:
        source_text = " ".join(source_blocks).lower()
        supported_card_types = {
            card_type
            for card_type in ("credit", "debit")
            if re.search(rf"\b{card_type}[\s-]+cards?\b", source_text)
        }
        if not requested_card_types.issubset(supported_card_types):
            return {"answer": ABSTENTION, "citations": [], "found": False}

    prompt = (
        "Answer the question using only the retrieved source text below. "
        "Do not add facts, conditions, dates, numbers, or policy conclusions not stated in that text. "
        "For numerical questions, distinguish the requested value from any condition or deadline, "
        "include the numerical answer when the sources state one, and use only the number "
        "that answers what was asked. "
        f"{_language_script_instruction(question)} "
        "Answer concisely and do not mix terms or values from different products or provisions. "
        "If the text does not answer the question, reply exactly: "
        f"{ABSTENTION}\n\n"
        f"Question:\n{question}\n\n"
        "Retrieved source text:\n"
        + "\n\n---\n\n".join(source_blocks)
        + "\n\nAnswer:"
    )
    payload = json.dumps(
        {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
        ensure_ascii=False,
    ).encode("utf-8")
    request = Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise OllamaRequestError(
            f"Ollama returned HTTP {error.code}: {detail or error.reason}"
        ) from error
    except (socket.timeout, TimeoutError) as error:
        raise OllamaTimeoutError(
            f"Ollama did not respond within {timeout:g} seconds."
        ) from error
    except URLError as error:
        if isinstance(error.reason, (socket.timeout, TimeoutError)):
            raise OllamaTimeoutError(
                f"Ollama did not respond within {timeout:g} seconds."
            ) from error
        raise OllamaConnectionError(
            f"Could not connect to Ollama at {OLLAMA_URL}: {error.reason}"
        ) from error

    answer = response_payload.get("response")
    if not isinstance(answer, str) or not answer.strip():
        raise OllamaRequestError("Ollama returned an empty or malformed response.")
    answer = answer.strip()
    if answer == ABSTENTION:
        return {"answer": answer, "citations": [], "found": False}
    if _answer_repeats_question(answer, question):
        return {"answer": ABSTENTION, "citations": [], "found": False}
    romanized_question = _is_romanized_hindi_question(question)
    answer_has_devanagari = bool(re.search(r"[\u0900-\u097f]", answer))
    answer_words = set(re.findall(r"[a-z]+", answer.lower()))
    if romanized_question and (
        answer_has_devanagari or not answer_words.intersection(ROMAN_HINDI_MARKERS)
    ):
        return {
            "answer": _fallback_for_question(source_blocks[0], question),
            "citations": citations,
            "found": True,
        }
    source_numbers = set().union(*(_numeric_values(text) for text in source_blocks))
    answer_numbers = _numeric_values(answer)
    answer_word_count = len(re.findall(r"\b[\w\u0900-\u097f]+\b", answer))
    missing_requested_number = (
        _question_requests_numeric_fact(question)
        and bool(source_numbers)
        and not answer_numbers
    )
    if (
        missing_requested_number
        or not answer_numbers.issubset(source_numbers)
        or not _has_source_term_support(answer, source_blocks)
        or answer_word_count > MAX_GENERATED_ANSWER_WORDS
    ):
        table_answer = _relevant_table_fallback(question, source_blocks)
        return {
            "answer": table_answer
            or _fallback_for_question(source_blocks[0], question),
            "citations": citations,
            "found": True,
        }
    return {"answer": answer, "citations": citations, "found": True}
