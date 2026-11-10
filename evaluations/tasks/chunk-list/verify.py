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
        from chunks import chunk_list
        for items, size, expected in [([], 2, []), ([1, 2, 3, 4, 5], 2, [[1, 2], [3, 4], [5]]), ([1, 2], 4, [[1, 2]]), ([1, 2], 1, [[1], [2]])]:
            original = items.copy()
            chunks = chunk_list(items, size)
            self.assertEqual(chunks, expected)
            if chunks:
                chunks[0].append(99)
            self.assertEqual(items, original)
        for size in [0, -1]:
            with self.assertRaises(ValueError):
                chunk_list([], size)
        for size in [True, 1.5, '2']:
            with self.assertRaises(TypeError):
                chunk_list([], size)


if __name__ == "__main__":
    unittest.main()
