import unittest

from scripts.export_paper_table import escape_latex, make_table


class TableTests(unittest.TestCase):
    def test_escaping(self):
        self.assertEqual(escape_latex("macro_f1 & 90%"), r"macro\_f1 \& 90\%")

    def test_table(self):
        out = make_table(
            [["metric", "value"], ["Macro_F1", "0.57"]],
            "Train & test",
            "tab:example",
        )
        self.assertIn(r"\caption{Train \& test}", out)
        self.assertIn(r"Macro\_F1", out)
        self.assertIn(r"\label{tab:example}", out)

    def test_ragged_rows(self):
        with self.assertRaises(ValueError):
            make_table([["a", "b"], ["single"]], "caption", "tab:x")

    def test_label_injection(self):
        with self.assertRaises(ValueError):
            make_table([["a"]], "caption", "x} \\input{secret}")


if __name__ == "__main__":
    unittest.main()
