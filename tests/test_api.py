import json
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from main import app
from scripts.answer_generation import (
    ABSTENTION,
    OllamaConnectionError,
    OllamaRequestError,
    OllamaTimeoutError,
)


class FakeOllamaResponse:
    def __init__(self, answer: str) -> None:
        self.payload = json.dumps({"response": answer}).encode("utf-8")

    def __enter__(self) -> "FakeOllamaResponse":
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def read(self) -> bytes:
        return self.payload


class AskEndpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)
        self.retrieval_result = {
            "status": "ok",
            "results": [
                {
                    "chunk_id": "chunk-tenure",
                    "source_pdf": "15_Grih_Rin_Niti_HL-2025-08.pdf",
                    "pages": [1],
                    "section_ids": ["home-loan-tenure"],
                    "snippet": "Maximum home-loan tenure is 30 years.",
                }
            ],
        }

    @patch("main.answer_question")
    @patch("main.retrieve")
    def test_ask_passes_retrieval_result_to_answer_generation(
        self, mock_retrieve: object, mock_answer: object
    ) -> None:
        mock_retrieve.return_value = self.retrieval_result
        mock_answer.return_value = {
            "answer": "Maximum tenure is 30 years.",
            "citations": [
                {
                    "source_pdf": "15_Grih_Rin_Niti_HL-2025-08.pdf",
                    "pages": [1],
                    "chunk_id": "chunk-tenure",
                    "section_ids": ["home-loan-tenure"],
                }
            ],
            "found": True,
        }

        response = self.client.post(
            "/ask", json={"question": "What is the maximum home-loan tenure?"}
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "answer": "Maximum tenure is 30 years.",
                "citations": [
                    {
                        "source_pdf": "15_Grih_Rin_Niti_HL-2025-08.pdf",
                        "pages": [1],
                        "chunk_id": "chunk-tenure",
                        "section_ids": ["home-loan-tenure"],
                    }
                ],
                "found": True,
            },
        )
        mock_retrieve.assert_called_once_with("What is the maximum home-loan tenure?")
        mock_answer.assert_called_once_with(
            "What is the maximum home-loan tenure?", self.retrieval_result
        )

    def test_invalid_and_blank_questions_return_422(self) -> None:
        for payload in ({}, {"question": ""}, {"question": "   "}, {"question": 123}):
            with self.subTest(payload=payload):
                response = self.client.post("/ask", json=payload)
                self.assertEqual(response.status_code, 422)

        response = self.client.post("/ask", content="not-json")
        self.assertEqual(response.status_code, 422)

    @patch("main.answer_question")
    @patch("main.retrieve")
    def test_no_evidence_returns_abstention_without_ollama_call(
        self, mock_retrieve: object, mock_answer: object
    ) -> None:
        mock_retrieve.return_value = {
            "status": "no_sufficient_evidence",
            "results": [],
        }
        mock_answer.return_value = {
            "answer": "I don't have sufficient evidence in the retrieved sources to answer that.",
            "citations": [],
            "found": False,
        }

        response = self.client.post(
            "/ask",
            json={"question": "What is the annual fee for the Bank's credit cards?"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()["found"])
        self.assertEqual(response.json()["citations"], [])
        mock_answer.assert_called_once_with(
            "What is the annual fee for the Bank's credit cards?",
            mock_retrieve.return_value,
        )

    @patch("scripts.answer_generation.urlopen")
    def test_real_pipeline_abstains_for_unsupported_product_question(
        self, mock_urlopen: object
    ) -> None:
        response = self.client.post(
            "/ask",
            json={"question": "What is the annual fee for the Bank's credit cards?"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {"answer": ABSTENTION, "citations": [], "found": False},
        )
        mock_urlopen.assert_not_called()

    @patch("scripts.answer_generation.urlopen")
    def test_real_pipeline_abstains_for_unrelated_domains(
        self, mock_urlopen: object
    ) -> None:
        questions = (
            "What is the weather in Delhi tomorrow?",
            "What is the current Tesla stock price?",
            "Who won the last football world cup?",
            "How do I bake sourdough bread?",
        )
        for question in questions:
            with self.subTest(question=question):
                response = self.client.post("/ask", json={"question": question})

                self.assertEqual(response.status_code, 200)
                self.assertEqual(
                    response.json(),
                    {"answer": ABSTENTION, "citations": [], "found": False},
                )
        mock_urlopen.assert_not_called()

    @patch("scripts.answer_generation.urlopen")
    def test_real_retrieval_pipeline_cites_supported_evidence(
        self, mock_urlopen: object
    ) -> None:
        question = "home loan ke liye maximum tenure kitna hai?"
        mock_urlopen.return_value = FakeOllamaResponse(
            "Home loan ki adhiktam avadhi 30 varsh hai."
        )

        response = self.client.post("/ask", json={"question": question})

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["found"])
        self.assertTrue(body["citations"])
        self.assertTrue(
            any(
                citation["source_pdf"] == "15_Grih_Rin_Niti_HL-2025-08.pdf"
                and "home-loan-tenure" in citation.get("section_ids", [])
                for citation in body["citations"]
            )
        )
        mock_urlopen.assert_called_once()

    @patch("scripts.answer_generation.urlopen")
    def test_real_pipeline_replaces_nonnumeric_tenure_hallucination(
        self, mock_urlopen: object
    ) -> None:
        mock_urlopen.return_value = FakeOllamaResponse("Thintin.")
        question = "home loan ke liye maximum tenure kitna hai?"

        response = self.client.post("/ask", json={"question": question})

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["answer"], "grih rin kii adhikatam avadhi 30 varsh hai.")
        self.assertTrue(body["found"])
        self.assertEqual(len(body["citations"]), 1)
        self.assertEqual(
            body["citations"][0]["section_ids"],
            ["home-loan-tenure"],
        )
        mock_urlopen.assert_called_once()

    @patch("scripts.answer_generation.urlopen")
    def test_real_pipeline_keeps_romanized_hindi_script(
        self, mock_urlopen: object
    ) -> None:
        mock_urlopen.return_value = FakeOllamaResponse(
            "गृह ऋण की अधिकतम अवधि 30 वर्ष है।"
        )
        question = "home loan ke liye maximum tenure kitna hai?"

        response = self.client.post("/ask", json={"question": question})

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["answer"], "grih rin kii adhikatam avadhi 30 varsh hai.")
        self.assertNotRegex(body["answer"], r"[\u0900-\u097f]")
        self.assertTrue(body["found"])
        mock_urlopen.assert_called_once()

    @patch("main.retrieve", side_effect=RuntimeError("retrieval failed"))
    def test_retrieval_failure_returns_503(self, _: object) -> None:
        response = self.client.post("/ask", json={"question": "A valid question"})
        self.assertEqual(response.status_code, 503)
        self.assertNotIn("answer", response.json())

    def test_ollama_failures_return_http_errors_without_answer(self) -> None:
        failures = (
            (OllamaConnectionError("not running"), 503),
            (OllamaTimeoutError("timed out"), 504),
            (OllamaRequestError("bad response"), 502),
        )
        for error, expected_status in failures:
            with self.subTest(error=type(error).__name__):
                with (
                    patch("main.retrieve", return_value=self.retrieval_result),
                    patch("main.answer_question", side_effect=error),
                ):
                    response = self.client.post(
                        "/ask", json={"question": "A valid question"}
                    )
                self.assertEqual(response.status_code, expected_status)
                self.assertNotIn("answer", response.json())

    def test_health_endpoint(self) -> None:
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})


if __name__ == "__main__":
    unittest.main()
