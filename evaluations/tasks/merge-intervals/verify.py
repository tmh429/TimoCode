"""评测器调用 python -I verify.py WORKSPACE，退出码 0 表示通过。"""

import copy
import importlib.util
from pathlib import Path
import sys
import unittest


def load_candidate(workspace):
    spec = importlib.util.spec_from_file_location("candidate_intervals", workspace / "intervals.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.merge_intervals


class MergeIntervalsTests(unittest.TestCase):
    def test_cases(self):
        cases = [
            ([], []),
            ([[2, 4]], [[2, 4]]),
            ([[5, 7], [1, 3], [2, 6]], [[1, 7]]),
            ([[1, 3], [3, 5]], [[1, 5]]),
            ([[1, 10], [2, 3], [4, 5]], [[1, 10]]),
            ([[1, 2], [1, 2]], [[1, 2]]),
            ([[-4, -1], [-2, 0], [3, 3]], [[-4, 0], [3, 3]]),
            ([[0, 0], [0, 2], [5, 6]], [[0, 2], [5, 6]]),
        ]
        for intervals, expected in cases:
            with self.subTest(intervals=intervals):
                original = copy.deepcopy(intervals)
                result = merge_intervals(intervals)
                self.assertEqual(result, expected)
                self.assertEqual(intervals, original)
                if result:
                    result[0][0] -= 100
                    self.assertEqual(intervals, original)


if __name__ == "__main__":
    merge_intervals = load_candidate(Path(sys.argv.pop(1)).resolve(strict=True))
    unittest.main()
