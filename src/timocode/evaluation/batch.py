"""先快照任务集，再串行评测，每次完成后保存批次进度与汇总。"""

import asyncio
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
import uuid

from pydantic import Field

from timocode.evaluation.runner import (
    EvaluationStatus, _load_task_inputs, _save_json, _verification_command, run_evaluation,
)
from timocode.evaluation.summary import EvaluationSummary, load_result, summarize_results
from timocode.llm.base import LLMProvider
from timocode.llm.schemas import DataModel


class BatchRun(DataModel):
    task_id: str
    run_id: str
    status: EvaluationStatus
    result_file: str


class BatchResult(DataModel):
    schema_version: Literal[1] = 1
    batch_id: str
    model: str
    docker_image: str
    image_id: str | None = None
    started_at: str
    state: Literal["running", "completed", "cancelled", "error"] = "running"
    task_ids: list[str]
    active_task_id: str | None = None
    runs: list[BatchRun] = Field(default_factory=list)
    summary: EvaluationSummary
    artifact_directory: str
    error: str | None = None


def discover_tasks(paths: Sequence[str | Path]) -> list[Path]:
    manifests = []
    for value in paths:
        path = Path(value).resolve(strict=True)
        if path.is_file():
            manifests.append(path)
        elif (path / "task.json").is_file():
            manifests.append(path / "task.json")
        else:
            children = sorted(path.glob("*/task.json"))
            if not children:
                raise ValueError(f"目录中没有任务清单。{path}")
            manifests.extend(children)
    if not manifests:
        raise ValueError("批量评测至少需要一个任务")
    return manifests


async def run_batch(
    paths: Sequence[str | Path], provider: LLMProvider, model: str, *,
    output_directory: str | Path = ".timocode/evaluations",
    docker_image: str = "python:3.12-slim", trace_secrets: tuple[str, ...] = (),
) -> BatchResult:
    """普通任务失败后继续，取消和不可保存的异常传播给调用者。"""
    if not model.strip():
        raise ValueError("模型名称不能为空")
    tasks = {}
    for manifest in discover_tasks(paths):
        task, verifier = _load_task_inputs(manifest)
        try:
            compile(verifier, str(manifest.parent / task.verifier), "exec")
        except SyntaxError as error:
            raise ValueError(f"验收脚本语法错误。{manifest}，第 {error.lineno} 行") from None
        _verification_command(verifier.decode("utf-8"), "0" * 32)
        if task.id in tasks:
            raise ValueError(f"任务 ID 重复。{task.id}")
        tasks[task.id] = (task, verifier)
    batch_id = uuid.uuid4().hex
    root = Path(output_directory).resolve() / f"batch-{batch_id}"
    root.mkdir(parents=True, exist_ok=False)
    batch = BatchResult(
        batch_id=batch_id, model=model, docker_image=docker_image,
        started_at=datetime.now(timezone.utc).isoformat(), task_ids=sorted(tasks),
        summary=summarize_results([]), artifact_directory=str(root),
    )
    results = []
    try:
        # 所有任务先落盘，后续评测不依赖源目录的并发修改。
        for task_id in batch.task_ids:
            task, verifier = tasks[task_id]
            destination = root / "tasks" / task_id
            destination.mkdir(parents=True)
            _save_json(destination / "task.json", task.model_copy(update={"verifier": "verify.py"}))
            (destination / "verify.py").write_bytes(verifier)
        _save_json(root / "batch.json", batch)
        for task_id in batch.task_ids:
            batch.active_task_id = task_id
            _save_json(root / "batch.json", batch)
            try:
                await run_evaluation(
                    root / "tasks" / task_id / "task.json", provider, model,
                    output_directory=root / "runs" / task_id,
                    docker_image=batch.image_id or docker_image, trace_secrets=trace_secrets,
                )
            finally:
                # 取消时 Runner 仍会写入 cancelled 结果，在传播取消前纳入进度。
                files = list((root / "runs" / task_id).glob("*/result.json"))
                if len(files) == 1:
                    result = load_result(files[0])
                    if batch.image_id is None:
                        batch.image_id = result.image_id
                    results.append(result)
                    batch.runs.append(BatchRun(
                        task_id=task_id, run_id=result.run_id, status=result.status,
                        result_file=files[0].relative_to(root).as_posix(),
                    ))
                    batch.summary = summarize_results(results)
            if not batch.runs or batch.runs[-1].task_id != task_id:
                raise OSError(f"任务未保存唯一结果。{task_id}")
            batch.active_task_id = None
            _save_json(root / "batch.json", batch)
        batch.state = "completed"
    except asyncio.CancelledError:
        batch.state = "cancelled"
        batch.error = "批量运行已取消，已完成结果保留"
        raise
    except Exception as error:
        batch.state = "error"
        batch.error = f"{type(error).__name__}。{error}"
        raise
    finally:
        _save_json(root / "batch.json", batch)
    return batch
