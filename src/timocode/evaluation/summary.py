"""从已有结果汇总指标，兼容缺少 metrics 的 Phase 6.2 记录。"""

from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from pydantic import Field

from timocode.evaluation.metrics import ToolCounts, ToolMetrics, measure_run, read_tool_metrics
from timocode.evaluation.runner import EvaluationResult
from timocode.llm.schemas import DataModel
from timocode.security.workspace import Workspace


class AggregateMetrics(DataModel):
    run_count: int
    task_count: int
    passed: int
    task_success_rate: float | None
    status_counts: dict[str, int]
    failure_counts: dict[str, int]
    artifact_error_runs: int
    average_elapsed_seconds: float | None
    average_iterations: float | None
    tool_metrics_runs: int
    tools: ToolMetrics
    tool_success_rate: float | None
    average_tool_calls: float | None
    token_complete_runs: int
    reported_input_tokens: int
    reported_output_tokens: int
    total_tokens: int | None
    average_complete_run_tokens: float | None


class EvaluationSummary(DataModel):
    schema_version: int = 1
    overall: AggregateMetrics
    by_model: dict[str, AggregateMetrics] = Field(default_factory=dict)
    by_task: dict[str, AggregateMetrics] = Field(default_factory=dict)


def _aggregate(results: Sequence[EvaluationResult]) -> AggregateMetrics:
    metrics = [measure_run(result, result.metrics.tools if result.metrics is not None else None) for result in results]
    tool_metrics = [metric.tools for metric in metrics if metric.tools is not None]
    by_name = {}
    total = ToolCounts()
    for tool_metric in tool_metrics:
        for name, counts in tool_metric.by_name.items():
            aggregate = by_name.setdefault(name, ToolCounts())
            for field in ToolCounts.model_fields:
                setattr(aggregate, field, getattr(aggregate, field) + getattr(counts, field))
                setattr(total, field, getattr(total, field) + getattr(counts, field))
    complete_tokens = [metric.total_tokens for metric in metrics if metric.token_usage_complete]
    passed = sum(result.status == "passed" for result in results)
    return AggregateMetrics(
        run_count=len(results), task_count=len({result.task_id for result in results}), passed=passed,
        task_success_rate=passed / len(results) if results else None,
        status_counts=dict(sorted(Counter(result.status for result in results).items())),
        failure_counts=dict(sorted(Counter(metric.failure_type for metric in metrics if metric.failure_type is not None).items())),
        artifact_error_runs=sum(result.artifact_error is not None for result in results),
        average_elapsed_seconds=sum(result.elapsed_seconds for result in results) / len(results) if results else None,
        average_iterations=sum(result.iterations for result in results) / len(results) if results else None,
        tool_metrics_runs=len(tool_metrics), tools=ToolMetrics(total=total, by_name=by_name),
        tool_success_rate=total.succeeded / total.attempted if total.attempted else None,
        average_tool_calls=total.attempted / len(tool_metrics) if tool_metrics else None,
        token_complete_runs=len(complete_tokens),
        reported_input_tokens=sum(metric.reported_input_tokens for metric in metrics),
        reported_output_tokens=sum(metric.reported_output_tokens for metric in metrics),
        total_tokens=sum(complete_tokens) if len(complete_tokens) == len(results) else None,
        average_complete_run_tokens=sum(complete_tokens) / len(complete_tokens) if complete_tokens else None,
    )


def summarize_results(results: Sequence[EvaluationResult]) -> EvaluationSummary:
    """每个唯一运行计一次，所有状态均进入任务成功率分母。"""
    if len({result.run_id for result in results}) != len(results):
        raise ValueError("汇总包含重复 run_id，请移除重复运行记录")
    models = sorted({result.model for result in results})
    tasks = sorted({result.task_id for result in results})
    return EvaluationSummary(
        overall=_aggregate(results),
        by_model={model: _aggregate([result for result in results if result.model == model]) for model in models},
        by_task={task: _aggregate([result for result in results if result.task_id == task]) for task in tasks},
    )


def load_result(path: str | Path) -> EvaluationResult:
    """以结果文件的实际位置查找旧追踪，允许移动整个产物目录。"""
    path = Path(path)
    result = EvaluationResult.model_validate_json(path.read_text(encoding="utf-8"))
    tools = result.metrics.tools if result.metrics is not None else None
    if tools is None and result.session_id is not None:
        candidates = []
        for trace in sorted((path.parent / "traces").glob("*.jsonl")):
            counts = read_tool_metrics(trace, result.session_id)
            if counts is not None:
                candidates.append(counts)
        tools = candidates[0] if len(candidates) == 1 else None
    result.metrics = measure_run(result, tools)
    return result


def _experiment_result_files(directory: Path) -> list[Path]:
    # 延迟导入避免实验调度与汇总模块的初始化依赖形成环。
    from timocode.evaluation.batch import BatchResult
    from timocode.evaluation.experiment import ExperimentResult

    experiment = ExperimentResult.model_validate_json((directory / "experiment.json").read_text(encoding="utf-8"))
    workspace = Workspace(directory)
    files = []
    for reference in experiment.batches:
        batch_file = workspace.resolve_path(reference.batch_file)
        batch = BatchResult.model_validate_json(batch_file.read_text(encoding="utf-8"))
        batch_workspace = Workspace(batch_file.parent)
        files.extend(batch_workspace.resolve_path(run.result_file) for run in batch.runs)
    return files


def summarize_paths(paths: Sequence[str | Path]) -> EvaluationSummary:
    """接受结果 JSON、单次、批次、实验目录及其父目录。"""
    files = []
    for value in paths:
        path = Path(value).resolve(strict=True)
        if path.is_file():
            files.append(path)
        elif (path / "result.json").is_file():
            files.append(path / "result.json")
        elif (path / "batch.json").is_file():
            files.extend(sorted(path.glob("runs/*/*/result.json")))
        elif (path / "experiment.json").is_file():
            files.extend(_experiment_result_files(path))
        else:
            discovered = sorted(path.glob("*/result.json"))
            discovered.extend(sorted(path.glob("batch-*/runs/*/*/result.json")))
            for index in sorted(path.glob("*/experiment.json")):
                discovered.extend(_experiment_result_files(index.parent))
            if not discovered:
                raise ValueError(f"目录中没有已完成的评测结果。{path}")
            files.extend(discovered)
    return summarize_results([load_result(path) for path in files])
