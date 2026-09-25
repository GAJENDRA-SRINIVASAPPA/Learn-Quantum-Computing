import unittest

from app import local_explanation, simulate


class QuantumMentorTests(unittest.TestCase):
    def test_bell_state_is_correlated(self):
        result = simulate("H 0\nCNOT 0 1")
        self.assertEqual(result["probabilities"], {"00": 0.5, "11": 0.5})
        self.assertIn("correlated", local_explanation(result))

    def test_x_flips_second_qubit(self):
        self.assertEqual(simulate("X 1")["probabilities"], {"01": 1.0})

    def test_invalid_gate_is_explained(self):
        with self.assertRaises(ValueError):
            simulate("Y 0")


if __name__ == "__main__":
    unittest.main()
