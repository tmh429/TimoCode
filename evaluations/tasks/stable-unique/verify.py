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
        from unique import stable_unique
        for values, expected in [([], []), ([3, 1, 3, 2, 1], [3, 1, 2]), (['b', 'a', 'b'], ['b', 'a']), ([None, 2, None, 1], [None, 2, 1])]:
            original = values.copy()
            output = stable_unique(values)
            self.assertEqual(output, expected)
            self.assertIsNot(output, values)
            self.assertEqual(values, original)


if __name__ == "__main__":
    unittest.main()
