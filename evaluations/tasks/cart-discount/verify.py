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
        from shop.discounts import apply_discount
        from shop.cart import total
        self.assertEqual(apply_discount(19.99, 15), 16.99)
        self.assertEqual(apply_discount(20, 100), 0)
        items = [{'price': 10, 'quantity': 2}, {'price': 5, 'quantity': 1}]
        original = copy.deepcopy(items)
        self.assertEqual(total(items, 20), 20)
        self.assertEqual(total(items), 25)
        self.assertEqual(total([], 10), 0)
        self.assertEqual(items, original)
        for percent in [-1, 101]:
            with self.assertRaises(ValueError):
                apply_discount(10, percent)
            with self.assertRaises(ValueError):
                total(items, percent)


if __name__ == "__main__":
    unittest.main()
