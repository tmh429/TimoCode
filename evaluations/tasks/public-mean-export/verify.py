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
        from statsmini import mean
        from statsmini.functions import mean as implementation
        self.assertIs(mean, implementation)
        for values, expected in [([1, 2, 3], 2), ([-2, 3], 0.5), ((1.5, 2.5), 2), ([4], 4)]:
            original = copy.deepcopy(values)
            self.assertEqual(mean(values), expected)
            self.assertEqual(values, original)
        with self.assertRaises(ValueError):
            mean([])


if __name__ == "__main__":
    unittest.main()
