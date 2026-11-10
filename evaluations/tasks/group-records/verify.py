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
        from records import group_records
        rows = [{'team': 'b', 'id': 1}, {'team': 'a', 'id': 2}, {'team': 'b', 'id': 3}]
        original = copy.deepcopy(rows)
        grouped = group_records(rows, 'team')
        self.assertEqual(list(grouped), ['b', 'a'])
        self.assertEqual(grouped, {'b': [rows[0], rows[2]], 'a': [rows[1]]})
        self.assertIs(grouped['b'][0], rows[0])
        self.assertEqual(rows, original)
        self.assertEqual(group_records([], 'team'), {})
        self.assertEqual(group_records([{'k': None}], 'k'), {None: [{'k': None}]})
        with self.assertRaises(KeyError):
            group_records([{}], 'missing')


if __name__ == "__main__":
    unittest.main()
