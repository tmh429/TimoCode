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
        self.assertEqual(answer, {'enabled': True, 'max_jobs': 2})
        self.assertIs(type(answer['enabled']), bool)
        self.assertIs(type(answer['max_jobs']), int)

    def test_original_files_unchanged(self):
        originals = {'settings.py': 'import json\n'
                        'from pathlib import Path\n'
                        '\n'
                        'def load(environment):\n'
                        "    base = json.loads(Path('config/defaults.json').read_text())\n"
                        '    '
                        "base.update(json.loads(Path(f'config/{environment}.json').read_text()))\n"
                        '    return base\n',
         'config/defaults.json': '{"enabled": false, "max_jobs": 2}\n',
         'config/production.json': '{"enabled": true}\n',
         'config/development.json': '{"enabled": false, "max_jobs": 8}\n'}
        for name, content in originals.items():
            self.assertEqual((WORKSPACE / name).read_text(encoding='utf-8'), content)


if __name__ == "__main__":
    unittest.main()
