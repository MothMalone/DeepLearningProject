import unittest

from src.vn_gpt2_math.answers import clean_model_output, parse_number, vote_candidate_texts
from src.vn_gpt2_math.targets import VIET_DIGIT_MAP, clean_math_text


class AnswerUtilityTests(unittest.TestCase):
    def test_clean_model_output_strips_units_and_hue(self):
        self.assertEqual(clean_model_output("####đáp án là: 72 ô tô"), "####đáp án là: 72")
        self.assertEqual(clean_model_output("####đáp án là: 5huehue"), "####đáp án là: 5")
        self.assertEqual(clean_model_output("####đáp án là: 3.5kg"), "####đáp án là: 3.5")
        self.assertEqual(clean_model_output("####đáp án là: -4"), "####đáp án là: -4")

    def test_clean_model_output_preserves_supported_latex(self):
        self.assertEqual(clean_model_output("####đáp án là: \\frac{1}{2}"), "####đáp án là: \\frac{1}{2}")
        self.assertEqual(clean_model_output("####đáp án là: 50\\sqrt{10}"), "####đáp án là: 50\\sqrt{10}")
        self.assertEqual(clean_model_output("####đáp án là: 14\\frac{6}{7}"), "####đáp án là: 14\\frac{6}{7}")

    def test_parse_number_latex(self):
        self.assertAlmostEqual(parse_number("\\frac{1}{2}"), 0.5)
        self.assertAlmostEqual(parse_number("50\\sqrt{10}"), 50 * (10**0.5))
        self.assertEqual(parse_number("không"), 0.0)

    def test_common_non_numeric_words_are_not_digits(self):
        self.assertNotIn("bay", VIET_DIGIT_MAP)
        self.assertNotIn("không", VIET_DIGIT_MAP)
        self.assertEqual(VIET_DIGIT_MAP["bảy"], 7)

    def test_candidate_voting_prefers_numeric_majority(self):
        chosen, debug = vote_candidate_texts(
            [
                "####đáp án là: 37",
                "48 - 11 = 37\n####đáp án là: 37 ô",
                "####đáp án là: 38",
            ]
        )
        self.assertEqual(chosen, "####đáp án là: 37")
        self.assertEqual(debug["selection_reason"], "numeric_majority")
        self.assertEqual(debug["selected_vote_count"], 2)

    def test_clean_math_text_preserves_fraction_grouping(self):
        text = clean_math_text(r"\frac{2+4-8+16+32-64}{4+8-16+32+64-128}")
        self.assertEqual(text, "((2+4-8+16+32-64)/(4+8-16+32+64-128))")

    def test_clean_math_text_preserves_symbolic_math_signals(self):
        text = clean_math_text(r"\sum_{1 \le a < b < c} \frac{1}{2^a 3^b 5^c} + \cos x + \pi")
        self.assertIn("<=", text)
        self.assertIn("((1)/(2^a 3^b 5^c))", text)
        self.assertIn("cos x", text)
        self.assertIn("pi", text)

    def test_clean_math_text_strips_asy_blocks(self):
        text = clean_math_text("Hình vẽ [asy] pair A=(0,10); draw(A); [/asy] còn lại 12")
        self.assertNotIn("asy", text.lower())
        self.assertNotIn("(0.10)", text)
        self.assertIn("còn lại 12", text)

    def test_clean_math_text_handles_escaped_currency(self):
        self.assertEqual(clean_math_text(r"$\$130$"), "130")
        self.assertEqual(clean_math_text(r"\$2"), "2")


if __name__ == "__main__":
    unittest.main()
