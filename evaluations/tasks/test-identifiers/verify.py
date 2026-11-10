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
        import shutil
        import subprocess
        import tempfile
        implementations = ['import keyword\n'
         'import re\n'
         '\n'
         'def is_identifier(text):\n'
         "    return bool(re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', text)) and not "
         'keyword.iskeyword(text)\n',
         'import keyword\n'
         'import re\n'
         '\n'
         'def is_identifier(text):\n'
         "    return bool(re.fullmatch(r'[A-Za-z0-9_]+', text)) and not "
         'keyword.iskeyword(text)\n',
         'import re\n'
         '\n'
         'def is_identifier(text):\n'
         "    return bool(re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', text))\n",
         'import keyword\n'
         'import re\n'
         '\n'
         'def is_identifier(text):\n'
         "    return bool(re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*', text)) and not "
         'keyword.iskeyword(text)\n']
        harness = ('import sys\n'
         'import unittest\n'
         '\n'
         "sys.path.insert(0, '.')\n"
         "suite = unittest.defaultTestLoader.discover('tests')\n"
         'result = unittest.TextTestRunner().run(suite)\n'
         'passed = suite.countTestCases() >= 3 and result.wasSuccessful()\n'
         'sys.exit(0 if passed else 1)\n')
        for index, implementation in enumerate(implementations):
            with tempfile.TemporaryDirectory() as directory:
                candidate = Path(directory) / 'project'
                shutil.copytree(WORKSPACE, candidate)
                (candidate / 'validators.py').write_text(implementation, encoding='utf-8')
                result = subprocess.run([sys.executable, '-I', '-c', harness], cwd=candidate, capture_output=True, text=True, timeout=5)
                if index == 0:
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                else:
                    self.assertEqual(result.returncode, 1, '测试未能发现缺陷版本 ' + str(index))

    def test_original_files_unchanged(self):
        originals = {'validators.py': 'import keyword\n'
                          'import re\n'
                          '\n'
                          'def is_identifier(text):\n'
                          "    return bool(re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', text)) and "
                          'not keyword.iskeyword(text)\n',
         'tests/__init__.py': ''}
        for name, content in originals.items():
            self.assertEqual((WORKSPACE / name).read_text(encoding='utf-8'), content)


if __name__ == "__main__":
    unittest.main()
