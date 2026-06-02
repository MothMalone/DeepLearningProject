import unittest

from src.vn_gpt2_math.metrics import evaluate


class MetricTests(unittest.TestCase):
    def test_evaluate_exact_numeric_answer(self):
        preds = [{"id": 0, "query_vi": "q", "type": "GSM_Rephrased", "model_output": "####đáp án là: 42"}]
        gold = [{"id": 0, "query_vi": "q", "type": "GSM_Rephrased", "response_vi": "Câu trả lời là: 42"}]
        report = evaluate(preds, gold)
        self.assertEqual(report["summary"]["raw_score"], 10)
        self.assertEqual(report["summary"]["exact_count"], 1)

    def test_evaluate_unit_tail_cleanup(self):
        preds = [{"id": 0, "query_vi": "q", "type": "GSM_Rephrased", "model_output": "####đáp án là: 72 ô"}]
        gold = [{"id": 0, "query_vi": "q", "type": "GSM_Rephrased", "response_vi": "Đáp án là: 72"}]
        report = evaluate(preds, gold)
        self.assertEqual(report["summary"]["raw_score"], 10)


if __name__ == "__main__":
    unittest.main()
