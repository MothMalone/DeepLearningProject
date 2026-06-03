import unittest

from src.vn_gpt2_math.dedup import DedupConfig, canonicalize_query, deduplicate_records


class DedupTests(unittest.TestCase):
    def test_canonicalize_query_normalizes_surface_noise(self):
        left = canonicalize_query("Nếu Lan có 1,5 kg táo thì còn lại bao nhiêu?")
        right = canonicalize_query("neu lan co 1.5 kg tao thi con lai bao nhieu")
        self.assertEqual(left, right)

    def test_exact_duplicate_keeps_one_record(self):
        records = [
            {"type": "GSM_Rephrased", "query_vi": "Lan có 5 quả táo. Hỏi còn lại bao nhiêu?", "_target": "####đáp án là: 5"},
            {"type": "GSM_AnsAug", "query_vi": "Lan có 5 quả táo. Hỏi còn lại bao nhiêu?", "_target": "####đáp án là: 5"},
        ]
        kept, report = deduplicate_records(records, DedupConfig(similar=True))
        self.assertEqual(len(kept), 1)
        self.assertEqual(report["dropped_exact"], 1)
        self.assertEqual(kept[0]["type"], "GSM_Rephrased")

    def test_similar_template_with_different_numbers_is_preserved(self):
        records = [
            {"type": "GSM_Rephrased", "query_vi": "Lan có 5 quả táo và cho đi 2 quả. Lan còn bao nhiêu quả?", "_target": "####đáp án là: 3"},
            {"type": "GSM_Rephrased", "query_vi": "Lan có 7 quả táo và cho đi 2 quả. Lan còn bao nhiêu quả?", "_target": "####đáp án là: 5"},
        ]
        kept, report = deduplicate_records(records, DedupConfig(similar=True))
        self.assertEqual(len(kept), 2)
        self.assertEqual(report["dropped_similar"], 0)


if __name__ == "__main__":
    unittest.main()
