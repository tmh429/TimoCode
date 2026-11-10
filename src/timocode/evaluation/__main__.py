"""单次、批量评测与已有结果的离线汇总入口。"""

import argparse
import asyncio
import json
from pathlib import Path

from timocode.evaluation.runner import _save_json, run_evaluation
from timocode.evaluation.batch import run_batch
from timocode.evaluation.summary import summarize_paths
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


async def _run_batch(arguments: argparse.Namespace) -> int:
    settings = load_settings(arguments.env_file)
    async with OpenAICompatibleProvider(settings) as provider:
        batch = await run_batch(
            arguments.batch, provider, settings.model,
            output_directory=arguments.output_dir, docker_image=arguments.docker_image,
            trace_secrets=(settings.api_key.get_secret_value(),),
        )
    print(json.dumps(batch.model_dump(mode="json"), indent=2))
    return 0 if batch.summary.overall.passed == len(batch.task_ids) else 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Coding Task 的单次、批量评测与离线汇总")
    parser.add_argument("task", type=Path, nargs="?", help="任务 JSON 清单")
    parser.add_argument("--env-file", type=Path, help="模型配置文件")
    parser.add_argument("--docker-image", default="python:3.12-slim", help="本地 Linux Python 镜像")
    parser.add_argument("--output-dir", type=Path, default=Path(".timocode/evaluations"))
    parser.add_argument("--summarize", type=Path, nargs="+", help="汇总已有结果文件或产物目录")
    parser.add_argument("--summary-output", type=Path, help="另存汇总 JSON，需要 --summarize")
    parser.add_argument("--batch", type=Path, nargs="+", help="串行执行任务清单或任务目录")
    arguments = parser.parse_args()
    if sum(value is not None for value in (arguments.task, arguments.summarize, arguments.batch)) != 1:
        parser.error("任务路径、--batch 和 --summarize 必须选择其中一个")
    if arguments.summary_output is not None and arguments.summarize is None:
        parser.error("--summary-output 需要 --summarize")
    try:
        if arguments.summarize is not None:
            summary = summarize_paths(arguments.summarize)
            if arguments.summary_output is not None:
                if arguments.summary_output.exists():
                    raise ValueError("汇总输出文件已存在，请使用新路径")
                arguments.summary_output.parent.mkdir(parents=True, exist_ok=True)
                _save_json(arguments.summary_output, summary)
            print(json.dumps(summary.model_dump(mode="json"), indent=2))
            code = 0
        elif arguments.batch is not None:
            code = asyncio.run(_run_batch(arguments))
        else:
            code = asyncio.run(_run(arguments))
    except (KeyboardInterrupt, asyncio.CancelledError):
        parser.exit(130, "评测已取消\n")
    except (ValueError, OSError) as error:
        parser.exit(2, f"评测配置错误。{error}\n")
    raise SystemExit(code)


if __name__ == "__main__":
    main()
