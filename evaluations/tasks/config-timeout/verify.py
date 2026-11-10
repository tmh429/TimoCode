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
        from service.settings import load_settings
        from service.cli import render_config
        for env, timeout in [({}, 30), ({'APP_TIMEOUT': '12'}, 12), ({'APP_TIMEOUT': '0'}, 30), ({'APP_TIMEOUT': '-3'}, 30), ({'APP_TIMEOUT': 'bad'}, 30)]:
            expected = {'host': 'localhost', 'timeout': timeout}
            self.assertEqual(load_settings(env), expected)
            self.assertEqual(json.loads(render_config(env)), expected)
        self.assertEqual(json.loads(render_config({'APP_HOST': 'example', 'APP_TIMEOUT': '5'})), {'host': 'example', 'timeout': 5})


if __name__ == "__main__":
    unittest.main()
