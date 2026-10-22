import json
import socket
import unittest
from unittest.mock import patch
from urllib.error import URLError

from scripts.answer_generation import (
    ABSTENTION,
    OLLAMA_MODEL,
    OllamaConnectionError,
    OllamaTimeoutError,
    answer_question,
)


class FakeResponse:
    def __init__(self, payload: dict[str, str]) -> None:
        self.payload = json.dumps(payload).encode("utf-8")

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def read(self) -> bytes:
        return self.payload


class AnswerGenerationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.retrieval_result = {
            "status": "ok",
            "results": [
                {
                    "chunk_id": "doc-15-home-loan-tenure-p1-03-8c8000b6a0",
                    "source_pdf": "15_Grih_Rin_Niti_HL-2025-08.pdf",
                    "pages": [1],
                    "section_ids": ["home-loan-tenure"],
                    "snippet": "The maximum home-loan tenure is 30 years.",
                    "metadata": {"status_assessment": "not_assessed"},
                }
            ],
        }

    @patch("scripts.answer_generation.urlopen")
    def test_supported_question_returns_answer_and_citation_from_retrieved_metadata(
        self, mock_urlopen: object
    ) -> None:
        mock_urlopen.return_value = FakeResponse({"response": "The maximum tenure is 30 years."})

        result = answer_question("What is the maximum home-loan tenure?", self.retrieval_result)

        self.assertEqual(result["answer"], "The maximum tenure is 30 years.")
        self.assertTrue(result["found"])
        self.assertEqual(
            result["citations"],
            [
                {
                    "source_pdf": "15_Grih_Rin_Niti_HL-2025-08.pdf",
                    "pages": [1],
                    "chunk_id": "doc-15-home-loan-tenure-p1-03-8c8000b6a0",
                    "section_ids": ["home-loan-tenure"],
                }
            ],
        )
        request = mock_urlopen.call_args.args[0]
        payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual(payload["model"], OLLAMA_MODEL)
        self.assertIn("What is the maximum home-loan tenure?", payload["prompt"])
        self.assertIn("The maximum home-loan tenure is 30 years.", payload["prompt"])
        self.assertIn("Write the answer in English using the Latin alphabet", payload["prompt"])
        self.assertNotIn("not_assessed", payload["prompt"])

    @patch("scripts.answer_generation.urlopen")
    def test_unsupported_credit_card_fee_abstains_without_calling_ollama(
        self, mock_urlopen: object
    ) -> None:
        result = answer_question(
            "What is the annual fee for the Bank's credit cards?",
            {"status": "no_sufficient_evidence", "results": []},
        )

        self.assertEqual(result, {"answer": ABSTENTION, "citations": [], "found": False})
        mock_urlopen.assert_not_called()

    @patch("scripts.answer_generation.urlopen")
    def test_credit_card_question_does_not_use_debit_card_fee_evidence(
        self, mock_urlopen: object
    ) -> None:
        debit_fee_evidence = {
            "status": "ok",
            "results": [
                {
                    "chunk_id": "debit-card-fee",
                    "source_pdf": "03_Savings_Account_Product_Guide.pdf",
                    "pages": [1],
                    "section_ids": ["savings-service-charges"],
                    "source_text": "Service: Debit card annual fee | Classic Rs. 250; Platinum Rs. 750.",
                }
            ],
        }

        result = answer_question(
            "What is the annual fee for the Bank's credit cards?",
            debit_fee_evidence,
        )

        self.assertEqual(result, {"answer": ABSTENTION, "citations": [], "found": False})
        mock_urlopen.assert_not_called()

    @patch("scripts.answer_generation.urlopen")
    def test_empty_retrieval_results_abstain_without_calling_ollama(
        self, mock_urlopen: object
    ) -> None:
        result = answer_question(
            "What is the maximum home-loan tenure?",
            {"status": "ok", "results": []},
        )

        self.assertEqual(result, {"answer": ABSTENTION, "citations": [], "found": False})
        mock_urlopen.assert_not_called()

    @patch("scripts.answer_generation.urlopen")
    def test_citations_preserve_each_source_page_and_section(
        self, mock_urlopen: object
    ) -> None:
        mock_urlopen.return_value = FakeResponse({"response": "The source supports this answer."})
        retrieval_result = {
            "status": "ok",
            "results": [
                {
                    "chunk_id": "chunk-one",
                    "source_pdf": "01_Fixed_Deposit_Scheme_Circular_FD-2025-04.pdf",
                    "pages": [1, 2],
                    "section_ids": ["fd-premature-withdrawal"],
                    "snippet": "The withdrawal condition continues across both pages.",
                },
                {
                    "chunk_id": "chunk-two",
                    "source_pdf": "02_FD_Interest_Rate_Revision_FD-2026-02.pdf",
                    "pages": [1],
                    "section_ids": ["fd-revised-withdrawal-penalty"],
                    "snippet": "Revised premature withdrawal penalties.",
                },
            ],
        }

        result = answer_question("What are the withdrawal conditions?", retrieval_result)

        self.assertEqual(
            result["citations"],
            [
                {
                    "source_pdf": "01_Fixed_Deposit_Scheme_Circular_FD-2025-04.pdf",
                    "pages": [1, 2],
                    "chunk_id": "chunk-one",
                    "section_ids": ["fd-premature-withdrawal"],
                },
                {
                    "source_pdf": "02_FD_Interest_Rate_Revision_FD-2026-02.pdf",
                    "pages": [1],
                    "chunk_id": "chunk-two",
                    "section_ids": ["fd-revised-withdrawal-penalty"],
                },
            ],
        )

    @patch("scripts.answer_generation.urlopen")
    def test_answer_prompt_uses_full_retrieved_text_not_truncated_snippet(
        self, mock_urlopen: object
    ) -> None:
        mock_urlopen.return_value = FakeResponse({"response": "3.50 प्रतिशत वार्षिक।"})
        source_text = (
            "खाते में दैनिक शेष राशि | ब्याज दर (वार्षिक)\n"
            "1 लाख रुपये तक | 2.75 प्रतिशत\n"
            "1 लाख रुपये से अधिक और 10 लाख रुपये तक | 3.00 प्रतिशत\n"
            "10 लाख रुपये से अधिक | 3.50 प्रतिशत\n"
            "01 अप्रैल 2026 से प्रभावी।"
        )
        retrieval_result = {
            "status": "ok",
            "results": [
                {
                    "chunk_id": "savings-rate",
                    "source_pdf": "12_Bachat_Khata_Byaj_Dar_Suchna_SB-2026-03.pdf",
                    "pages": [1],
                    "section_ids": ["savings-revised-rates-hindi"],
                    "snippet": source_text[:80],
                    "source_text": source_text,
                }
            ],
        }

        result = answer_question(
            "10 ला ख रुपये से अधि क शेष वा ले बचत खा ते पर ब्या ज दर क्या है?",
            retrieval_result,
        )

        request = mock_urlopen.call_args.args[0]
        payload = json.loads(request.data.decode("utf-8"))
        self.assertIn("10 लाख रुपये से अधिक | 3.50 प्रतिशत", payload["prompt"])
        self.assertIn("01 अप्रैल 2026 से प्रभावी", payload["prompt"])
        self.assertIn("Write the answer in Hindi using Devanagari script", payload["prompt"])
        self.assertEqual(result["answer"], "3.50 प्रतिशत वार्षिक।")
        self.assertTrue(result["found"])
        self.assertEqual(
            result["citations"][0]["source_pdf"],
            "12_Bachat_Khata_Byaj_Dar_Suchna_SB-2026-03.pdf",
        )

    @patch("scripts.answer_generation.urlopen")
    def test_long_answer_to_rate_table_question_falls_back_to_matching_row(
        self, mock_urlopen: object
    ) -> None:
        source_text = (
            "MT\nमेरिडियन ट्रस्ट बैंक लिमिटेड\nप्रधान कार्यालय\n"
            "परिपत्र संख्या SB/2026/03\nदिनांक 15 मार्च 2026\n"
            "विषय बचत खाते पर ब्याज दरों में संशोधन\n"
            "सभी ग्राहकों को सूचित किया जाता है कि 01 अप्रैल 2026 से बचत खातों पर "
            "ब्याज दरें निम्नानुसार होंगी। यह सूचना पिछली सूचना SB/2025/06 का स्थान लेती है।\n"
            "खाते में दैनिक शेष राशि | ब्याज दर (वार्षिक)\n"
            "1 लाख रुपये तक | 2.75 प्रतिशत\n"
            "1 लाख रुपये से अधिक और 10 लाख रुपये तक | 3.00 प्रतिशत\n"
            "10 लाख रुपये से अधिक | 3.50 प्रतिशत"
        )
        mock_urlopen.return_value = FakeResponse(
            {
                "response": (
                    "MT मेरिडियन ट्रस्ट बैंक लिमिटेड प्रधान कार्यालय परिपत्र संख्या "
                    "SB/2026/03 दिनांक 15 मार्च 2026 विभाग खुदरा देयता विभाग विषय "
                    "बचत खाते पर ब्याज दरों में संशोधन सभी ग्राहकों को सूचित किया जाता "
                    "है कि 01 अप्रैल 2026 से बचत खातों पर ब्याज दरें निम्नानुसार होंगी। "
                    "यह सूचना पिछली सूचना SB/2025/06 का स्थान लेती है। खाते में दैनिक "
                    "शेष राशि ब्याज दर वार्षिक 1 लाख रुपये तक 2.75 प्रतिशत 1 लाख रुपये "
                    "से अधिक और 10 लाख रुपये तक 3.00 प्रतिशत 10 लाख रुपये से अधिक "
                    "3.50 प्रतिशत।"
                )
            }
        )
        retrieval_result = {
            "status": "ok",
            "results": [
                {
                    "chunk_id": "savings-revised-rates",
                    "source_pdf": "12_Bachat_Khata_Byaj_Dar_Suchna_SB-2026-03.pdf",
                    "pages": [1],
                    "section_ids": ["savings-revised-rates-hindi"],
                    "source_text": source_text,
                },
                {
                    "chunk_id": "savings-interest-credit",
                    "source_pdf": "12_Bachat_Khata_Byaj_Dar_Suchna_SB-2026-03.pdf",
                    "pages": [1],
                    "section_ids": ["savings-interest-credit-hindi"],
                    "source_text": "ब्याज त्रैमासिक आधार पर खाते में जमा किया जाएगा।",
                },
            ],
        }

        result = answer_question(
            "10 लाख रुपये से अधिक शेष वाले बचत खाते पर ब्याज दर क्या है?",
            retrieval_result,
        )

        self.assertEqual(
            result["answer"],
            "ब्याज दर (वार्षिक): 3.50 प्रतिशत 10 लाख रुपये से अधिक "
            "(01 अप्रैल 2026 से)।",
        )
        self.assertTrue(result["found"])
        self.assertEqual(
            [citation["chunk_id"] for citation in result["citations"]],
            ["savings-revised-rates"],
        )

    @patch("scripts.answer_generation.urlopen")
    def test_answer_context_excludes_tenures_from_other_products(
        self, mock_urlopen: object
    ) -> None:
        mock_urlopen.return_value = FakeResponse(
            {"response": "Home loan ki adhiktam avadhi 30 varsh (30 years) hai."}
        )
        retrieval_result = {
            "status": "ok",
            "results": [
                {
                    "chunk_id": "home-tenure",
                    "source_pdf": "15_Grih_Rin_Niti_HL-2025-08.pdf",
                    "pages": [1],
                    "section_ids": ["home-loan-tenure"],
                    "score": 0.43,
                    "source_text": "3. ऋण अवधि\nगृह ऋण की अधिकतम अवधि 30 वर्ष है।",
                },
                {
                    "chunk_id": "personal-loan-tenure",
                    "source_pdf": "07_Personal_Loan_Product_Note_PL-2026-01.pdf",
                    "pages": [1],
                    "section_ids": ["personal-loan-amount-tenure"],
                    "score": 0.25,
                    "source_text": "Tenure: 12 to 60 months.",
                },
            ],
        }

        result = answer_question(
            "home loan ke liye maximum tenure kitna hai?", retrieval_result
        )

        payload = json.loads(mock_urlopen.call_args.args[0].data.decode("utf-8"))
        self.assertIn("गृह ऋण की अधिकतम अवधि 30 वर्ष है", payload["prompt"])
        self.assertNotIn("12 to 60 months", payload["prompt"])
        self.assertEqual(result["answer"], "Home loan ki adhiktam avadhi 30 varsh (30 years) hai.")
        self.assertEqual(
            [citation["source_pdf"] for citation in result["citations"]],
            ["15_Grih_Rin_Niti_HL-2025-08.pdf"],
        )
        self.assertTrue(result["found"])

    @patch("scripts.answer_generation.urlopen")
    def test_romanised_hindi_question_requests_latin_script_and_specific_citation(
        self, mock_urlopen: object
    ) -> None:
        mock_urlopen.return_value = FakeResponse(
            {"response": "Home loan ki adhiktam avadhi 30 varsh (30 years) hai."}
        )
        retrieval_result = {
            "status": "ok",
            "results": [
                {
                    "chunk_id": "home-tenure",
                    "source_pdf": "15_Grih_Rin_Niti_HL-2025-08.pdf",
                    "pages": [1],
                    "section_ids": ["home-loan-tenure"],
                    "score": 0.43,
                    "source_text": "3. ऋण अवधि\nगृह ऋण की अधिकतम अवधि 30 वर्ष है।",
                },
                {
                    "chunk_id": "home-charges",
                    "source_pdf": "15_Grih_Rin_Niti_HL-2025-08.pdf",
                    "pages": [1],
                    "section_ids": ["home-loan-charges"],
                    "score": 0.30,
                    "source_text": "Processing fee is 0.35 percent.",
                },
                {
                    "chunk_id": "personal-tenure",
                    "source_pdf": "07_Personal_Loan_Product_Note_PL-2026-01.pdf",
                    "pages": [1],
                    "section_ids": ["personal-loan-amount-tenure"],
                    "score": 0.25,
                    "source_text": "Personal loan tenure is 12 to 60 months.",
                },
            ],
        }

        result = answer_question(
            "home loan ke liye maximum tenure kitna hai?", retrieval_result
        )

        payload = json.loads(mock_urlopen.call_args.args[0].data.decode("utf-8"))
        self.assertIn(
            "Write the answer in Romanised Hindi using the Latin alphabet, not Devanagari",
            payload["prompt"],
        )
        self.assertNotIn("Processing fee is 0.35 percent.", payload["prompt"])
        self.assertNotIn("Personal loan tenure is 12 to 60 months.", payload["prompt"])
        self.assertEqual(len(result["citations"]), 1)
        self.assertEqual(result["citations"][0]["section_ids"], ["home-loan-tenure"])
        self.assertTrue(result["found"])

    @patch("scripts.answer_generation.urlopen")
    def test_answer_context_keeps_explicitly_requested_sections(
        self, mock_urlopen: object
    ) -> None:
        mock_urlopen.return_value = FakeResponse(
            {"response": "The source describes both account variants and service charges."}
        )
        source_pdf = "03_Savings_Account_Service_Charges.pdf"
        retrieval_result = {
            "status": "ok",
            "results": [
                {
                    "chunk_id": "account-variants",
                    "source_pdf": source_pdf,
                    "pages": [1],
                    "section_ids": ["savings-account-variants"],
                    "source_text": "Account variants include Regular Savings and other options.",
                },
                {
                    "chunk_id": "service-charge-tiers",
                    "source_pdf": source_pdf,
                    "pages": [2],
                    "section_ids": ["regular-savings-service-charge-tiers"],
                    "source_text": "Regular Savings service-charge tiers depend on the MAB.",
                },
                {
                    "chunk_id": "unrelated-account-eligibility",
                    "source_pdf": source_pdf,
                    "pages": [2],
                    "section_ids": ["account-eligibility"],
                    "source_text": "Eligibility requires identity documents.",
                },
            ],
        }

        result = answer_question(
            "What are the savings account variants and regular savings service charges?",
            retrieval_result,
        )

        prompt = json.loads(mock_urlopen.call_args.args[0].data.decode("utf-8"))["prompt"]
        self.assertIn("Account variants include Regular Savings", prompt)
        self.assertIn("service-charge tiers depend on the MAB", prompt)
        self.assertNotIn("Eligibility requires identity documents", prompt)
        self.assertEqual(
            [citation["chunk_id"] for citation in result["citations"]],
            ["account-variants", "service-charge-tiers"],
        )

    @patch("scripts.answer_generation.urlopen")
    def test_unsupported_generated_number_falls_back_to_verified_source_text(
        self, mock_urlopen: object
    ) -> None:
        mock_urlopen.return_value = FakeResponse(
            {"response": "तीन महीने की देयता समय सीमा लागू होती है।"}
        )
        source_text = (
            "4. विलंब दंड - किराया देय तिथि से 30 दिन के भीतर जमा न करने पर "
            "बकाया किराये पर 10 प्रतिशत दंड शुल्क लगाया जाएगा।"
        )
        retrieval_result = {
            "status": "ok",
            "results": [
                {
                    "chunk_id": "doc-16-locker-late-penalty-p1-04-6cff203634",
                    "source_pdf": "16_Locker_Suvidha_Niyam_LOC-2016-07.pdf",
                    "pages": [1],
                    "section_ids": ["locker-late-penalty"],
                    "source_text": source_text,
                }
            ],
        }

        result = answer_question(
            "लॉकर का किराया देय तिथि से 30 दिन के भीतर जमा न करने पर कितना दंड लगता है?",
            retrieval_result,
        )

        prompt = json.loads(mock_urlopen.call_args.args[0].data.decode("utf-8"))["prompt"]
        self.assertIn("distinguish the requested value from any condition or deadline", prompt)
        self.assertEqual(
            result["answer"],
            "किराया देय तिथि से 30 दिन के भीतर जमा न करने पर "
            "बकाया किराये पर 10 प्रतिशत दंड शुल्क लगाया जाएगा।",
        )
        self.assertNotIn("\n", result["answer"])
        self.assertFalse(result["answer"].startswith("4."))
        self.assertTrue(result["found"])
        self.assertEqual(
            result["citations"][0]["section_ids"],
            ["locker-late-penalty"],
        )

    @patch("scripts.answer_generation.urlopen")
    def test_hindi_number_word_not_in_evidence_falls_back_to_source_text(
        self, mock_urlopen: object
    ) -> None:
        mock_urlopen.return_value = FakeResponse({"response": "तेरह"})
        source_text = (
            "4. विलंब दंड - किराया देय तिथि से 30 दिन के भीतर जमा न करने पर "
            "बकाया किराये पर 10 प्रतिशत दंड शुल्क लगाया जाएगा।"
        )
        retrieval_result = {
            "status": "ok",
            "results": [
                {
                    "chunk_id": "doc-16-locker-late-penalty-p1-04-6cff203634",
                    "source_pdf": "16_Locker_Suvidha_Niyam_LOC-2016-07.pdf",
                    "pages": [1],
                    "section_ids": ["locker-late-penalty"],
                    "source_text": source_text,
                }
            ],
        }

        result = answer_question(
            "लॉकर का किराया देय तिथि से 30 दिन के भीतर जमा न करने पर कितना दंड लगता है?",
            retrieval_result,
        )

        self.assertEqual(
            result["answer"],
            "किराया देय तिथि से 30 दिन के भीतर जमा न करने पर "
            "बकाया किराये पर 10 प्रतिशत दंड शुल्क लगाया जाएगा।",
        )
        self.assertTrue(result["found"])
        self.assertEqual(
            result["citations"][0]["chunk_id"],
            "doc-16-locker-late-penalty-p1-04-6cff203634",
        )

    @patch("scripts.answer_generation.urlopen")
    def test_numeric_fragment_without_source_terms_falls_back_to_source_text(
        self, mock_urlopen: object
    ) -> None:
        mock_urlopen.return_value = FakeResponse({"response": "अर्थमें 10"})
        source_text = (
            "4. विलंब दंड - किराया देय तिथि से 30 दिन के भीतर जमा न करने पर "
            "बकाया किराये पर 10 प्रतिशत दंड शुल्क लगाया जाएगा।"
        )
        retrieval_result = {
            "status": "ok",
            "results": [
                {
                    "chunk_id": "doc-16-locker-late-penalty-p1-04-6cff203634",
                    "source_pdf": "16_Locker_Suvidha_Niyam_LOC-2016-07.pdf",
                    "pages": [1],
                    "section_ids": ["locker-late-penalty"],
                    "source_text": source_text,
                }
            ],
        }

        result = answer_question(
            "लॉकर का किराया देय तिथि से 30 दिन के भीतर जमा न करने पर कितना दंड लगता है?",
            retrieval_result,
        )

        self.assertEqual(
            result["answer"],
            "किराया देय तिथि से 30 दिन के भीतर जमा न करने पर "
            "बकाया किराये पर 10 प्रतिशत दंड शुल्क लगाया जाएगा।",
        )
        self.assertTrue(result["found"])
        self.assertEqual(
            result["citations"][0]["section_ids"],
            ["locker-late-penalty"],
        )

    @patch("scripts.answer_generation.urlopen")
    def test_quantitative_question_missing_numeric_value_falls_back_to_source(
        self, mock_urlopen: object
    ) -> None:
        mock_urlopen.return_value = FakeResponse({"response": "Thintin."})
        retrieval_result = {
            "status": "ok",
            "results": [
                {
                    "chunk_id": "doc-15-home-loan-tenure-p1-03-8c8000b6a0",
                    "source_pdf": "15_Grih_Rin_Niti_HL-2025-08.pdf",
                    "pages": [1],
                    "section_ids": ["home-loan-tenure"],
                    "source_text": "3. ऋण अवधि\nगृह ऋण की अधिकतम अवधि 30 वर्ष है।",
                }
            ],
        }

        result = answer_question(
            "home loan ke liye maximum tenure kitna hai?",
            retrieval_result,
        )

        self.assertEqual(result["answer"], "grih rin kii adhikatam avadhi 30 varsh hai.")
        self.assertTrue(result["found"])
        self.assertEqual(
            result["citations"][0]["section_ids"],
            ["home-loan-tenure"],
        )

    @patch("scripts.answer_generation.urlopen")
    def test_devanagari_model_answer_for_romanized_question_uses_roman_script_fallback(
        self, mock_urlopen: object
    ) -> None:
        mock_urlopen.return_value = FakeResponse(
            {"response": "गृह ऋण की अधिकतम अवधि 30 वर्ष है।"}
        )
        retrieval_result = {
            "status": "ok",
            "results": [
                {
                    "chunk_id": "doc-15-home-loan-tenure-p1-03-8c8000b6a0",
                    "source_pdf": "15_Grih_Rin_Niti_HL-2025-08.pdf",
                    "pages": [1],
                    "section_ids": ["home-loan-tenure"],
                    "source_text": "3. ऋण अवधि\nगृह ऋण की अधिकतम अवधि 30 वर्ष है।",
                }
            ],
        }

        result = answer_question(
            "home loan ke liye maximum tenure kitna hai?",
            retrieval_result,
        )

        self.assertEqual(result["answer"], "grih rin kii adhikatam avadhi 30 varsh hai.")
        self.assertNotRegex(result["answer"], r"[\u0900-\u097f]")
        self.assertTrue(result["found"])

    @patch("scripts.answer_generation.urlopen")
    def test_english_model_answer_for_romanized_question_uses_roman_hindi_fallback(
        self, mock_urlopen: object
    ) -> None:
        mock_urlopen.return_value = FakeResponse(
            {"response": "The maximum home loan tenure is 30 years."}
        )
        retrieval_result = {
            "status": "ok",
            "results": [
                {
                    "chunk_id": "doc-15-home-loan-tenure-p1-03-8c8000b6a0",
                    "source_pdf": "15_Grih_Rin_Niti_HL-2025-08.pdf",
                    "pages": [1],
                    "section_ids": ["home-loan-tenure"],
                    "source_text": "3. ऋण अवधि\nगृह ऋण की अधिकतम अवधि 30 वर्ष है।",
                }
            ],
        }

        result = answer_question(
            "home loan ke liye maximum tenure kitna hai?",
            retrieval_result,
        )

        self.assertEqual(result["answer"], "grih rin kii adhikatam avadhi 30 varsh hai.")
        self.assertNotIn("maximum", result["answer"])

    @patch("scripts.answer_generation.urlopen")
    def test_model_echoing_question_is_returned_as_abstention(
        self, mock_urlopen: object
    ) -> None:
        question = "What is the maximum home-loan tenure?"
        mock_urlopen.return_value = FakeResponse({"response": question})

        result = answer_question(question, self.retrieval_result)

        self.assertEqual(result, {"answer": ABSTENTION, "citations": [], "found": False})

    @patch("scripts.answer_generation.urlopen", side_effect=URLError("connection refused"))
    def test_connection_failure_raises_without_returning_an_answer(
        self, mock_urlopen: object
    ) -> None:
        with self.assertRaises(OllamaConnectionError):
            answer_question("Question", self.retrieval_result)
        mock_urlopen.assert_called_once()

    @patch("scripts.answer_generation.urlopen", side_effect=socket.timeout("timed out"))
    def test_timeout_raises_without_returning_an_answer(self, mock_urlopen: object) -> None:
        with self.assertRaises(OllamaTimeoutError):
            answer_question("Question", self.retrieval_result, timeout=0.5)
        mock_urlopen.assert_called_once()


if __name__ == "__main__":
    unittest.main()
