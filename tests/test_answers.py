import unittest

from src.vn_gpt2_math.answers import clean_model_output, parse_number
from src.vn_gpt2_math.targets import VIET_DIGIT_MAP


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

    def test_bay_is_not_digit(self):
        self.assertNotIn("bay", VIET_DIGIT_MAP)
        self.assertEqual(VIET_DIGIT_MAP["bảy"], 7)


if __name__ == "__main__":
    unittest.main()
