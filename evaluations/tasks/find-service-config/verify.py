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
        answer = json.loads((WORKSPACE / 'answer.json').read_text())
        self.assertEqual(answer, {'host': 'internal.local', 'port': 8087, 'source': 'config/production.toml'})
        self.assertIs(type(answer['port']), int)

    def test_original_files_unchanged(self):
        originals = {'main.py': 'from pathlib import Path\n'
                    'import tomllib\n'
                    '\n'
                    "CONFIG = Path('config/production.toml')\n"
                    'def settings():\n'
                    "    return tomllib.loads(CONFIG.read_text())['service']\n",
         'config/production.toml': '[service]\nhost = "internal.local"\nport = 8087\n',
         'config/development.toml': '[service]\nhost = "localhost"\nport = 8080\n',
         'README.txt': '旧部署曾经使用端口 9000，请以 main.py 为准。\n'}
        for name, content in originals.items():
            self.assertEqual((WORKSPACE / name).read_text(encoding='utf-8'), content)


if __name__ == "__main__":
    unittest.main()
