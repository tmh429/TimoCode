"""python -m timocode.evaluation TASK_JSON 单任务评测入口。"""

import argparse
import asyncio
import json
from pathlib import Path

from timocode.evaluation.runner import run_evaluation
from timocode.llm.config import load_settings
from timocode.llm.provider import OpenAICompatibleProvider


async def _run(arguments: argparse.Namespace) -> int:
    settings = load_settings(arguments.env_file)
    async with OpenAICompatibleProvider(settings) as provider:
        result = await run_evaluation(
            arguments.task, provider, settings.model,
            output_directory=arguments.output_dir, docker_image=arguments.docker_image,
            trace_secrets=(settings.api_key.get_secret_value(),),
        )
    print(json.dumps(result.model_dump(mode="json"), indent=2))
    return 0 if result.status == "passed" else 1


def main() -> None:
    parser = argparse.ArgumentParser(description="在独立 Docker 工作区执行一个 Coding Task")
    parser.add_argument("task", type=Path, help="任务 JSON 清单")
    parser.add_argument("--env-file", type=Path, help="模型配置文件")
    parser.add_argument("--docker-image", default="python:3.12-slim", help="本地 Linux Python 镜像")
    parser.add_argument("--output-dir", type=Path, default=Path(".timocode/evaluations"))
    arguments = parser.parse_args()
    try:
        code = asyncio.run(_run(arguments))
    except (KeyboardInterrupt, asyncio.CancelledError):
        parser.exit(130, "评测已取消\n")
    except (ValueError, OSError) as error:
        parser.exit(2, f"评测配置错误。{error}\n")
    raise SystemExit(code)


if __name__ == "__main__":
    main()
