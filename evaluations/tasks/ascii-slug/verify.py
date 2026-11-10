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
        from slug import slugify
        for text, expected in [(' Hello, World! ', 'hello-world'), ('Résumé déjà vu', 'resume-deja-vu'), ('a___b---c', 'a-b-c'), ('中文', ''), ('', ''), ('Version 2.0', 'version-2-0')]:
            self.assertEqual(slugify(text), expected)


if __name__ == "__main__":
    unittest.main()
