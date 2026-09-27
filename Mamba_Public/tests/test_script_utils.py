import argparse
import tempfile
import unittest
from pathlib import Path

from changedetection.script.script_utils import parse_bool, read_name_list


class ScriptUtilsTest(unittest.TestCase):
    def test_parse_bool_handles_explicit_false(self):
        self.assertFalse(parse_bool("false"))
        self.assertFalse(parse_bool("0"))
        self.assertTrue(parse_bool("true"))
        self.assertTrue(parse_bool("1"))

    def test_parse_bool_rejects_unknown_value(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_bool("maybe")

    def test_read_name_list_ignores_blank_lines(self):
        with tempfile.TemporaryDirectory() as directory:
            list_path = Path(directory) / "train_set.txt"
            list_path.write_text("sample_a\n\n sample_b \n", encoding="utf-8")
            self.assertEqual(read_name_list(str(list_path)), ["sample_a", "sample_b"])

    def test_read_name_list_rejects_empty_file(self):
        with tempfile.TemporaryDirectory() as directory:
            list_path = Path(directory) / "empty.txt"
            list_path.write_text("\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                read_name_list(str(list_path))


if __name__ == "__main__":
    unittest.main()
