import unittest
from unittest.mock import patch

from scripts.retrieval import HybridRetriever, RetrievalConfig, SentenceTransformer


class RetrievalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.retriever = HybridRetriever()

    def assert_expected_result(self, question: str, expected_prefix: str, section_ids: set[str]) -> None:
        result = self.retriever.retrieve(question, top_k=5, min_score=0.18)
        self.assertEqual(result["status"], "ok", question)
        self.assertTrue(
            any(
                item["source_pdf"].startswith(expected_prefix)
                and set(item["section_ids"]) & section_ids
                for item in result["results"]
            ),
            msg=f"Expected source {expected_prefix} and section {section_ids} together in {result}",
        )

    def test_assignment_question_1_english_ltv(self) -> None:
        self.assert_expected_result(
            "What is the maximum loan-to-value ratio for a loan against debt mutual fund units?",
            "04_",
            {"las-eligible-securities-ltv"},
        )

    def test_assignment_question_2_hindi_rate(self) -> None:
        self.assert_expected_result(
            "10 लाख रुपये से अधिक शेष वाले बचत खाते पर ब्याज दर क्या है?",
            "12_",
            {"savings-revised-rates-hindi"},
        )

    def test_spaced_devanagari_savings_query_retains_full_rate_table_text(self) -> None:
        result = self.retriever.retrieve(
            "10 ला ख रुपये से अधि क शेष वा ले बचत खा ते पर ब्या ज दर क्या है?",
            top_k=5,
            min_score=0.18,
        )
        matching = next(
            item
            for item in result["results"]
            if item["source_pdf"].startswith("12_")
            and "savings-revised-rates-hindi" in item["section_ids"]
        )
        self.assertIn("10 लाख रुपये से अधिक | 3.50 प्रतिशत", matching["source_text"])
        self.assertIn("01 अप्रैल 2026", matching["source_text"])
        self.assertLessEqual(len(matching["snippet"]), 400)

    def test_assignment_question_3_romanised_hindi_tenure(self) -> None:
        self.assert_expected_result(
            "home loan ke liye maximum tenure kitna hai?",
            "15_",
            {"home-loan-tenure"},
        )

    def test_assignment_question_4_hindi_locker_penalty(self) -> None:
        self.assert_expected_result(
            "लॉकर का किराया देय तिथि से 30 दिन के भीतर जमा न करने पर कितना दंड लगता है?",
            "16_",
            {"locker-late-penalty"},
        )

    def test_assignment_question_5_unsupported_credit_card_fee_abstains(self) -> None:
        result = self.retriever.retrieve("What is the annual fee for the Bank's credit cards?", top_k=5, min_score=0.18)
        self.assertEqual(result["status"], "no_sufficient_evidence")
        self.assertEqual(result["results"], [])

    def test_paraphrases_and_exact_conditions(self) -> None:
        paraphrases = [
            ("What is the maximum LTV for debt mutual fund collateral?", "04_", {"las-eligible-securities-ltv"}),
            ("Maximum tenure for home loan?", "15_", {"home-loan-tenure"}),
            ("What percentage interest on savings balances above 10 lakh from 1 April 2026?", "12_", {"savings-revised-rates-hindi"}),
            ("What is the penalty if locker rent is not paid within 30 days?", "16_", {"locker-late-penalty"}),
        ]
        for question, prefix, sections in paraphrases:
            with self.subTest(question=question):
                self.assert_expected_result(question, prefix, sections)

    def test_document_3_variant_and_service_charge_evidence(self) -> None:
        result = self.retriever.retrieve(
            "Which savings account variant has a metro MAB of Rs. 5,000 and what is the charge if the MAB is not maintained?",
            top_k=5,
            min_score=0.18,
        )
        self.assertEqual(result["status"], "ok")
        section_ids = {item["section_ids"][0] if item["section_ids"] else "" for item in result["results"]}
        self.assertTrue({"savings-variants", "savings-service-charges"} & section_ids)

    def test_document_6_reporting_time_and_liabilities(self) -> None:
        result = self.retriever.retrieve(
            "For 4 to 7 working days, what is the liability for other savings accounts?",
            top_k=5,
            min_score=0.18,
        )
        self.assertEqual(result["status"], "ok")
        self.assertTrue(
            any(
                "grievance-electronic-transaction-liability" in item["section_ids"]
                for item in result["results"]
            )
        )

    def test_fixed_deposit_penalty_and_exception(self) -> None:
        result = self.retriever.retrieve(
            "What is the premature withdrawal penalty above Rs. 5,00,000 and when is there no penalty?",
            top_k=5,
            min_score=0.18,
        )
        self.assertEqual(result["status"], "ok")
        self.assertTrue(
            any(
                item["chunk_id"].startswith("doc-02-fd-revised-withdrawal-penalty")
                or item["chunk_id"].startswith("doc-01-fd-premature-withdrawal")
                for item in result["results"]
            )
        )

    def test_locker_revision_and_unchanged_rules_preserved(self) -> None:
        result = self.retriever.retrieve(
            "How much is the annual rent for a large locker from 1 April 2024 and which old rules still apply?",
            top_k=8,
            min_score=0.18,
        )
        self.assertEqual(result["status"], "ok")
        self.assertTrue(
            any(item["chunk_id"].startswith("doc-11-locker-revised-annual-rent") for item in result["results"])
        )
        self.assertTrue(
            any(item["chunk_id"].startswith("doc-16-locker-late-penalty") for item in result["results"])
        )

    def test_cross_page_clause_and_rule_10_retrieve_relevant_chapters(self) -> None:
        cases = [
            ("What is the premature withdrawal penalty and when is it waived on a depositor's death?", "01_", {"fd-premature-withdrawal"}),
            ("What is the liability limit for locker loss and natural disaster exception?", "16_", {"locker-bank-liability"}),
        ]
        for question, prefix, sections in cases:
            with self.subTest(question=question):
                self.assert_expected_result(question, prefix, sections)

    def test_unrelated_query_abstains(self) -> None:
        result = self.retriever.retrieve("What is the orbits of the moon?", top_k=5, min_score=0.18)
        self.assertEqual(result["status"], "no_sufficient_evidence")
        self.assertEqual(result["results"], [])

    def test_rankings_are_deterministic_and_metadata_is_cited(self) -> None:
        first = self.retriever.retrieve("What is the maximum loan-to-value ratio for a loan against debt mutual fund units?", top_k=5, min_score=0.18)
        second = self.retriever.retrieve("What is the maximum loan-to-value ratio for a loan against debt mutual fund units?", top_k=5, min_score=0.18)
        self.assertEqual([item["chunk_id"] for item in first["results"]], [item["chunk_id"] for item in second["results"]])
        self.assertIn("source_pdf", first["results"][0])
        self.assertIn("provenance", first["results"][0])

    def test_policy_relationship_metadata_is_preserved_in_retrieved_evidence(self) -> None:
        result = self.retriever.retrieve(
            "What is the annual rent for lockers from 1 April 2024?",
            top_k=5,
            min_score=0.18,
        )
        locker_revision = next(
            item
            for item in result["results"]
            if item["source_pdf"].startswith("11_")
            and "locker-revised-annual-rent" in item["section_ids"]
        )
        relationships = locker_revision["provenance"]["policy_relationships"]
        self.assertTrue(
            any(
                relation["relationship_id"] == "locker-annual-rent-revision"
                and relation["effective_date"] == "2024-04-01"
                for relation in relationships
            )
        )
        self.assertEqual(locker_revision["metadata"]["status_assessment"], "not_assessed")

    def test_semantic_disabled_by_default(self) -> None:
        retriever = HybridRetriever(config=RetrievalConfig(enable_semantic=False))
        result = retriever.retrieve(
            "What is the maximum loan-to-value ratio for a loan against debt mutual fund units?"
        )
        self.assertEqual(retriever.semantic_status, "disabled")
        self.assertFalse(result["used_semantic"])
        self.assertEqual(result["semantic_status"], "disabled")

    def test_embedding_package_unavailable_is_reported_and_lexical_score_is_not_penalized(self) -> None:
        lexical = HybridRetriever(config=RetrievalConfig(enable_semantic=False))
        with patch("scripts.retrieval.SentenceTransformer", None):
            semantic = HybridRetriever(config=RetrievalConfig(enable_semantic=True))
        question = "What is the maximum loan-to-value ratio for a loan against debt mutual fund units?"
        lexical_result = lexical.retrieve(question)
        fallback_result = semantic.retrieve(question)
        self.assertEqual(semantic.semantic_status, "unavailable")
        self.assertFalse(fallback_result["used_semantic"])
        self.assertIsNotNone(fallback_result["semantic_error"])
        self.assertEqual(
            [(item["chunk_id"], item["score"]) for item in lexical_result["results"]],
            [(item["chunk_id"], item["score"]) for item in fallback_result["results"]],
        )

    def test_model_load_failure_reports_fallback(self) -> None:
        with patch("scripts.retrieval.SentenceTransformer", side_effect=OSError("model files missing")):
            retriever = HybridRetriever(config=RetrievalConfig(enable_semantic=True))
        result = retriever.retrieve(
            "What is the maximum loan-to-value ratio for a loan against debt mutual fund units?"
        )
        self.assertEqual(retriever.semantic_status, "unavailable")
        self.assertIn("model files missing", result["semantic_error"])
        self.assertFalse(result["used_semantic"])
        self.assertTrue(result["results"])

    def test_semantic_chunk_embeddings_are_cached_and_rankings_are_deterministic(self) -> None:
        class FakeModel:
            def __init__(self) -> None:
                self.encode_calls: list[list[str]] = []

            def encode(self, texts: list[str], **_: object) -> list[list[float]]:
                self.encode_calls.append(list(texts))
                vectors = []
                for text in texts:
                    lowered = text.lower()
                    vectors.append(
                        [
                            float("loan" in lowered or "ऋण" in lowered),
                            float("locker" in lowered or "लॉकर" in lowered),
                            float("savings" in lowered or "बचत" in lowered),
                            1.0,
                        ]
                    )
                return vectors

        fake_model = FakeModel()
        with patch("scripts.retrieval.SentenceTransformer", return_value=fake_model):
            retriever = HybridRetriever(config=RetrievalConfig(enable_semantic=True))
        self.assertEqual(retriever.semantic_status, "loaded")
        chunk_encode_calls = len(fake_model.encode_calls)
        self.assertEqual(chunk_encode_calls, 1)

        question = "How much can I borrow using mutual fund units?"
        first = retriever.retrieve(question)
        after_first = len(fake_model.encode_calls)
        second = retriever.retrieve(question)
        self.assertEqual(after_first, chunk_encode_calls + 1)
        self.assertEqual(len(fake_model.encode_calls), after_first)
        self.assertEqual(
            [item["chunk_id"] for item in first["results"]],
            [item["chunk_id"] for item in second["results"]],
        )
        self.assertTrue(first["used_semantic"])
        unsupported = retriever.retrieve("What is the annual fee for the Bank's credit cards?")
        self.assertEqual(unsupported["status"], "no_sufficient_evidence")
        self.assertEqual(unsupported["results"], [])

    def test_semantic_encoding_failure_switches_to_reported_lexical_fallback(self) -> None:
        class BrokenModel:
            def encode(self, *_: object, **__: object) -> list[list[float]]:
                raise RuntimeError("encoder failed")

        with patch("scripts.retrieval.SentenceTransformer", return_value=BrokenModel()):
            retriever = HybridRetriever(config=RetrievalConfig(enable_semantic=True))
        result = retriever.retrieve(
            "What is the maximum loan-to-value ratio for a loan against debt mutual fund units?"
        )
        self.assertEqual(retriever.semantic_status, "unavailable")
        self.assertIn("encoder failed", result["semantic_error"])
        self.assertFalse(result["used_semantic"])
        self.assertTrue(result["results"])

    def test_semantic_model_integration_when_locally_available(self) -> None:
        if SentenceTransformer is None:
            self.skipTest("sentence-transformers is not installed")
        try:
            SentenceTransformer(
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                device="cpu",
                local_files_only=True,
            )
        except Exception as error:
            self.skipTest(f"multilingual model is not available locally: {error}")
        retriever = HybridRetriever(
            config=RetrievalConfig(enable_semantic=True, semantic_local_files_only=True)
        )
        self.assertEqual(retriever.semantic_status, "loaded")
        result = retriever.retrieve("10 लाख रुपये से अधिक शेष वाले बचत खाते पर ब्याज दर क्या है?")
        self.assertTrue(result["used_semantic"])
        self.assertTrue(
            any(
                item["source_pdf"].startswith("12_")
                and "savings-revised-rates-hindi" in item["section_ids"]
                for item in result["results"]
            )
        )
        romanised_result = retriever.retrieve(
            "ghar kharidne ke liye sabse lambi muddat kitni ho sakti hai?"
        )
        self.assertTrue(
            any(
                item["source_pdf"].startswith("15_")
                and "home-loan-tenure" in item["section_ids"]
                for item in romanised_result["results"]
            )
        )


if __name__ == "__main__":
    unittest.main()
