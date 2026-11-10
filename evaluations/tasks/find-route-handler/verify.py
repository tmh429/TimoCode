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
        self.assertEqual(json.loads((WORKSPACE / 'answer.json').read_text()), {'file': 'app/handlers/orders.py', 'function': 'create_order'})

    def test_original_files_unchanged(self):
        originals = {'app/__init__.py': '',
         'app/handlers/__init__.py': '',
         'app/routes.py': 'from .handlers.orders import create_order\n'
                          'from .handlers.health import health\n'
                          "ROUTES = {('GET', '/health'): health, ('POST', '/orders'): "
                          'create_order}\n',
         'app/handlers/orders.py': 'def create_order(payload):\n'
                                   "    return {'id': payload['id'], 'created': True}\n",
         'app/handlers/health.py': "def health(payload):\n    return {'ok': True}\n",
         'archive/orders.py': 'def create_order_v1(payload):\n    return payload\n'}
        for name, content in originals.items():
            self.assertEqual((WORKSPACE / name).read_text(encoding='utf-8'), content)


if __name__ == "__main__":
    unittest.main()
