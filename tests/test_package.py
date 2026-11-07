"""验证项目安装后，可以从项目目录之外导入。"""

import subprocess
import sys


def test_installed_package_is_importable_from_another_directory(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            "import timocode; "
            "from importlib.metadata import metadata; "
            "print(metadata('timocode')['Name'])",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "timocode"
