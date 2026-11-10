"""固定任务快照的配置对比实验，逐批记录可恢复的运行索引。"""

import asyncio
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Literal, Self
import uuid

from pydantic import Field, field_validator, model_validator

from timocode.evaluation.batch import BatchResult, _prepare_tasks, _snapshot_tasks, run_batch
from timocode.evaluation.options import EvaluationOptions
from timocode.evaluation.runner import _save_json
from timocode.llm.config import load_settings
from timocode.llm.provider import OpenAICompatibleProvider
from timocode.llm.schemas import DataModel


class ExperimentVariant(DataModel):
    name: str = Field(pattern=r"^[a-z0-9]+(?:[-_][a-z0-9]+)*$")
    model: str | None = Field(default=None, min_length=1)
    env_file: str | None = None
    options: EvaluationOptions = Field(default_factory=EvaluationOptions)

    @field_validator("model")
    @classmethod
    def nonblank_model(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("模型名称不能为空白")
        return value.strip() if value is not None else None


class ExperimentPlan(DataModel):
    schema_version: Literal[1] = 1
    name: str = Field(pattern=r"^[a-z0-9]+(?:[-_][a-z0-9]+)*$")
    tasks: list[str] = Field(min_length=1)
    variants: list[ExperimentVariant] = Field(min_length=2)
    repeats: int = Field(default=1, gt=0)
    docker_image: str = "python:3.12-slim"

    @field_validator("schema_version", mode="before")
    @classmethod
    def integer_version(cls, value: int) -> int:
        if type(value) is not int:
            raise ValueError("schema_version 必须是整数")
        return value

    @model_validator(mode="after")
    def unique_variants(self) -> Self:
        if len({variant.name for variant in self.variants}) != len(self.variants):
            raise ValueError("实验配置名称必须唯一")
        return self


class ExperimentBatch(DataModel):
    variant: str
    repeat: int = Field(ge=1)
    model: str
    provider_id: str
    batch_file: str
    state: Literal["running", "completed", "cancelled", "error"]


class ExperimentResult(DataModel):
    schema_version: Literal[1] = 1
    experiment_id: str
    plan: ExperimentPlan
    suite_sha256: str
    task_ids: list[str]
    state: Literal["running", "completed", "cancelled", "error"] = "running"
    active_variant: str | None = None
    active_repeat: int | None = None
    image_id: str | None = None
    batches: list[ExperimentBatch] = Field(default_factory=list)
    started_at: str
    artifact_directory: str
    error: str | None = None


def suite_fingerprint(tasks: dict) -> str:
    digest = hashlib.sha256()
    for task_id in sorted(tasks):
        task, verifier = tasks[task_id]
        canonical = task.model_copy(update={"verifier": "verify.py"}).model_dump(mode="json")
        digest.update(json.dumps(canonical, sort_keys=True, ensure_ascii=False).encode("utf-8"))
        digest.update(b"\0")
        digest.update(verifier)
        digest.update(b"\0")
    return digest.hexdigest()


async def run_experiment(
    plan_file: str | Path, *, env_file: str | Path | None = None,
    output_directory: str | Path = ".timocode/experiments",
) -> ExperimentResult:
    plan_file = Path(plan_file).resolve(strict=True)
    plan = ExperimentPlan.model_validate_json(plan_file.read_text(encoding="utf-8"))
    tasks = _prepare_tasks([plan_file.parent / value for value in plan.tasks])
    settings = {}
    for variant in plan.variants:
        config = plan_file.parent / variant.env_file if variant.env_file is not None else env_file
        loaded = load_settings(config)
        if variant.model is not None:
            loaded = loaded.model_copy(update={"model": variant.model})
        settings[variant.name] = loaded
    experiment_id = uuid.uuid4().hex
    root = Path(output_directory).resolve() / f"exp-{experiment_id[:12]}"
    root.mkdir(parents=True, exist_ok=False)
    experiment = ExperimentResult(
        experiment_id=experiment_id, plan=plan, suite_sha256=suite_fingerprint(tasks),
        task_ids=sorted(tasks), started_at=datetime.now(timezone.utc).isoformat(), artifact_directory=str(root),
    )
    try:
        manifests = _snapshot_tasks(tasks, root / "tasks")
        _save_json(root / "experiment.json", experiment)
        for repeat in range(1, plan.repeats + 1):
            for variant_index, variant in enumerate(plan.variants):
                experiment.active_variant = variant.name
                experiment.active_repeat = repeat
                _save_json(root / "experiment.json", experiment)
                config = settings[variant.name]
                destination = root / "b" / f"{repeat}-{variant_index}"
                try:
                    async with OpenAICompatibleProvider(config) as provider:
                        await run_batch(
                            manifests, provider, config.model, output_directory=destination,
                            docker_image=experiment.image_id or plan.docker_image,
                            trace_secrets=(config.api_key.get_secret_value(),), options=variant.options,
                            _compact_artifacts=True,
                        )
                finally:
                    files = list(destination.glob("batch-*/batch.json"))
                    if len(files) == 1:
                        batch = BatchResult.model_validate_json(files[0].read_text(encoding="utf-8"))
                        if experiment.image_id is None:
                            experiment.image_id = batch.image_id
                        experiment.batches.append(ExperimentBatch(
                            variant=variant.name, repeat=repeat, model=config.model,
                            provider_id=hashlib.sha256(config.base_url.encode("utf-8")).hexdigest(),
                            batch_file=files[0].relative_to(root).as_posix(), state=batch.state,
                        ))
                if len(files) != 1:
                    raise OSError("实验批次没有保存唯一索引")
                experiment.active_variant = None
                experiment.active_repeat = None
                _save_json(root / "experiment.json", experiment)
        experiment.state = "completed"
    except asyncio.CancelledError:
        experiment.state = "cancelled"
        experiment.error = "实验已取消，已完成批次保留"
        raise
    except Exception as error:
        experiment.state = "error"
        experiment.error = f"{type(error).__name__}。{error}"
        raise
    finally:
        _save_json(root / "experiment.json", experiment)
    return experiment
