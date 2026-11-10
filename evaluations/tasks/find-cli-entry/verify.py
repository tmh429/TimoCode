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
        self.assertEqual(json.loads((WORKSPACE / 'answer.json').read_text()), {'module': 'app.exporter', 'function': 'main', 'config_key': 'project.scripts.export'})

    def test_original_files_unchanged(self):
        originals = {'pyproject.toml': '[project]\n'
                           'name = "sample-export"\n'
                           'version = "0.1.0"\n'
                           '[project.scripts]\n'
                           'export = "app.exporter:main"\n'
                           'inspect = "app.inspect:main"\n',
         'app/__init__.py': '',
         'app/exporter.py': "def main():\n    print('export')\n",
         'app/inspect.py': "def main():\n    print('inspect')\n",
         'old_export.py': "def main():\n    print('obsolete')\n"}
        for name, content in originals.items():
            self.assertEqual((WORKSPACE / name).read_text(encoding='utf-8'), content)


if __name__ == "__main__":
    unittest.main()
