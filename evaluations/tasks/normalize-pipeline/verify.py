"""自包含标准库验收，首个参数为候选工作区。"""

import copy
import json
from pathlib import Path
import sys
import unittest

WORKSPACE = Path(sys.argv.pop(1)).resolve(strict=True)
sys.path.insert(0, str(WORKSPACE))


class TaskTests(unittest.TestCase):
    def test_behavior(self):
        from people.transform import parse_row
        from people.pipeline import process_rows
        rows = [{'name': ' Ada ', 'age': '20', 'extra': 'drop'}, {'name': ' ', 'age': '2'}, {'name': 'Bob', 'age': '-1'}, {'name': 'Cara', 'age': '0'}, {'name': 'Dan', 'age': 'bad'}, {}]
        original = copy.deepcopy(rows)
        self.assertEqual(parse_row(rows[0]), {'name': 'Ada', 'age': 20})
        for row in [rows[1], rows[2], rows[4], rows[5]]:
            self.assertIsNone(parse_row(row))
        self.assertEqual(process_rows(rows), [{'name': 'Ada', 'age': 20}, {'name': 'Cara', 'age': 0}])
        self.assertEqual(process_rows([]), [])
        self.assertEqual(rows, original)


if __name__ == "__main__":
    unittest.main()
