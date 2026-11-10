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
        from search import first_index
        for values, target, expected in [([], 1, -1), ([1, 2, 2, 2, 3], 2, 1), ([-3, -1, -1, 4], -1, 1), ([1, 2], 8, -1), ([4], 4, 0)]:
            original = values.copy()
            self.assertEqual(first_index(values, target), expected)
            self.assertEqual(values, original)
        class CountedList(list):
            reads = 0
            def __iter__(self):
                raise AssertionError('large input must use binary search')
            def index(self, *args):
                raise AssertionError('list.index is a linear scan')
            def __getitem__(self, index):
                self.reads += 1
                return super().__getitem__(index)
        values = CountedList(range(65536))
        self.assertEqual(first_index(values, 65000), 65000)
        self.assertLess(values.reads, 100)


if __name__ == "__main__":
    unittest.main()
