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
        from flags import parse_bool
        for text in ['true', ' YES ', '1', 'On']:
            self.assertIs(parse_bool(text), True)
        for text in ['false', ' NO ', '0', 'off']:
            self.assertIs(parse_bool(text), False)
        for text in ['', 'maybe', '2', ' true false ']:
            with self.assertRaises(ValueError):
                parse_bool(text)


if __name__ == "__main__":
    unittest.main()
