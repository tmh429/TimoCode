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
        from flatten import flatten_dict
        data = {'a': {'b': 1, 'c': {'d': 2}}, 'empty': {}, 'items': [1, 2], 'none': None}
        original = copy.deepcopy(data)
        self.assertEqual(flatten_dict(data), {'a.b': 1, 'a.c.d': 2, 'empty': {}, 'items': [1, 2], 'none': None})
        self.assertEqual(data, original)
        self.assertEqual(flatten_dict({}), {})


if __name__ == "__main__":
    unittest.main()
