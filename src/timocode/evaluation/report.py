"""校验实验条件与任务快照，生成配对比较和独立 HTML、CSV、图表。"""

import csv
import html
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import uuid

from pydantic import Field

from timocode.evaluation.batch import BatchResult, _prepare_tasks
from timocode.evaluation.experiment import ExperimentResult, suite_fingerprint
from timocode.evaluation.options import EvaluationOptions
from timocode.evaluation.runner import _save_json
from timocode.evaluation.summary import AggregateMetrics, load_result, summarize_results
from timocode.llm.schemas import DataModel
from timocode.security.workspace import Workspace


class ComparisonRow(DataModel):
    name: str
    models: list[str]
    provider_ids: list[str]
    options: EvaluationOptions
    planned_runs: int
    complete: bool
    aggregate: AggregateMetrics
    matched_runs: int = 0
    paired_wins: int = 0
    paired_losses: int = 0
    paired_ties: int = 0
    paired_success_delta: float | None = None
    paired_mean_token_delta: float | None = None
    paired_mean_latency_delta: float | None = None


class ComparisonReport(DataModel):
    experiment_id: str
    name: str
    state: str
    suite_sha256: str
    task_ids: list[str]
    repeats: int
    baseline: str
    image_id: str | None
    rows: list[ComparisonRow]
    runs: list[dict] = Field(default_factory=list)


def compare_experiment(directory: str | Path) -> ComparisonReport:
    root = Path(directory).resolve(strict=True)
    workspace = Workspace(root)
    experiment = ExperimentResult.model_validate_json((root / "experiment.json").read_text(encoding="utf-8"))
    tasks = _prepare_tasks([root / "tasks"])
    if suite_fingerprint(tasks) != experiment.suite_sha256 or sorted(tasks) != experiment.task_ids:
        raise ValueError("实验任务快照与指纹不一致")
    variants = {variant.name: variant for variant in experiment.plan.variants}
    results = {name: [] for name in variants}
    slots = {name: {} for name in variants}
    models = {name: set() for name in variants}
    providers = {name: set() for name in variants}
    seen_batches = set()
    seen_runs = set()
    records = []
    for reference in experiment.batches:
        key = (reference.variant, reference.repeat)
        if reference.variant not in variants or key in seen_batches or reference.repeat > experiment.plan.repeats:
            raise ValueError("实验批次条件重复或超出计划")
        seen_batches.add(key)
        batch_file = workspace.resolve_path(reference.batch_file)
        batch = BatchResult.model_validate_json(batch_file.read_text(encoding="utf-8"))
        variant = variants[reference.variant]
        if (batch.task_ids != experiment.task_ids or batch.options != variant.options
                or batch.model != reference.model or batch.state != reference.state):
            raise ValueError("批次任务或配置与实验条件不一致")
        if suite_fingerprint(_prepare_tasks([batch_file.parent / "tasks"])) != experiment.suite_sha256:
            raise ValueError("批次使用了不同的任务快照")
        models[variant.name].add(batch.model)
        providers[variant.name].add(reference.provider_id)
        batch_workspace = Workspace(batch_file.parent)
        for run in batch.runs:
            result_file = batch_workspace.resolve_path(run.result_file)
            result = load_result(result_file)
            slot = (run.task_id, reference.repeat)
            if run.task_id not in tasks or slot in slots[variant.name] or result.run_id in seen_runs:
                raise ValueError("实验包含未知任务或重复运行")
            if (result.task_id, result.run_id, result.status) != (run.task_id, run.run_id, run.status):
                raise ValueError("结果与批次索引不一致")
            if result.options != variant.options or result.model != batch.model:
                raise ValueError("结果的运行配置与条件不一致")
            if suite_fingerprint(_prepare_tasks([result_file.parent / "task"])) != suite_fingerprint({run.task_id: tasks[run.task_id]}):
                raise ValueError("单次运行使用了不同的任务快照")
            if experiment.image_id is not None and result.image_id is not None and result.image_id != experiment.image_id:
                raise ValueError("实验包含不同的 Docker 镜像 ID")
            seen_runs.add(result.run_id)
            slots[variant.name][slot] = result
            results[variant.name].append(result)
            context = result.agent_result.context_report if result.agent_result is not None else None
            records.append({
                "variant": variant.name, "repeat": reference.repeat, "task_id": result.task_id,
                "model": result.model, "status": result.status, "failure_type": result.metrics.failure_type,
                "elapsed_seconds": result.elapsed_seconds, "total_tokens": result.metrics.total_tokens,
                "tool_calls": result.metrics.tools.total.attempted if result.metrics.tools is not None else None,
                "error": result.error,
                "estimated_input": context.estimate.total if context is not None else None,
                "input_limit": context.input_limit if context is not None else None,
                "result_file": result_file.relative_to(root).as_posix(),
            })
    planned = len(tasks) * experiment.plan.repeats
    baseline = experiment.plan.variants[0].name
    rows = []
    for variant in experiment.plan.variants:
        if len(models[variant.name]) > 1 or len(providers[variant.name]) > 1:
            raise ValueError("同一实验条件包含不同模型或服务")
        matched = set(slots[baseline]) & set(slots[variant.name])
        wins = sum(slots[variant.name][slot].status == "passed" and slots[baseline][slot].status != "passed" for slot in matched)
        losses = sum(slots[baseline][slot].status == "passed" and slots[variant.name][slot].status != "passed" for slot in matched)
        token_deltas = [
            slots[variant.name][slot].metrics.total_tokens - slots[baseline][slot].metrics.total_tokens
            for slot in matched
            if slots[variant.name][slot].metrics.total_tokens is not None and slots[baseline][slot].metrics.total_tokens is not None
        ]
        complete = len(results[variant.name]) == planned
        if experiment.state == "completed" and not complete:
            raise ValueError("已完成实验缺少计划结果")
        rows.append(ComparisonRow(
            name=variant.name, models=sorted(models[variant.name]), provider_ids=sorted(providers[variant.name]), options=variant.options,
            planned_runs=planned, complete=complete, aggregate=summarize_results(results[variant.name]).overall,
            matched_runs=len(matched), paired_wins=wins, paired_losses=losses,
            paired_ties=len(matched) - wins - losses,
            paired_success_delta=(wins - losses) / len(matched) if matched else None,
            paired_mean_token_delta=sum(token_deltas) / len(matched) if matched and len(token_deltas) == len(matched) else None,
            paired_mean_latency_delta=(
                sum(slots[variant.name][slot].elapsed_seconds - slots[baseline][slot].elapsed_seconds for slot in matched) / len(matched)
                if matched else None
            ),
        ))
    return ComparisonReport(
        experiment_id=experiment.experiment_id, name=experiment.plan.name, state=experiment.state,
        suite_sha256=experiment.suite_sha256, task_ids=experiment.task_ids, repeats=experiment.plan.repeats,
        baseline=baseline, image_id=experiment.image_id, rows=rows, runs=records,
    )


def _chart(report: ComparisonReport, directory: Path) -> str:
    try:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.backends.backend_svg import FigureCanvasSVG
        from matplotlib.figure import Figure
    except ImportError:
        raise ValueError("生成图表需要安装评测依赖，请运行 python -m pip install -e '.[eval]'") from None
    figure = Figure(figsize=(12, 8), layout="constrained")
    figure.suptitle(f"{report.name} | tasks={len(report.task_ids)} | repeats={report.repeats}")
    panels = [
        ("Task success rate", [row.aggregate.task_success_rate for row in report.rows]),
        ("Total tokens (complete usage)", [row.aggregate.total_tokens for row in report.rows]),
        ("Mean task latency (seconds)", [row.aggregate.average_elapsed_seconds for row in report.rows]),
        ("Mean tool calls (known runs)", [row.aggregate.average_tool_calls for row in report.rows]),
    ]
    for axis, (title, values) in zip(figure.subplots(2, 2).flat, panels):
        positions = list(range(len(report.rows)))
        axis.bar(positions, [float("nan") if value is None else value for value in values], color="#4263eb")
        axis.set_xticks(positions, [row.name for row in report.rows], rotation=15)
        axis.set_title(title)
        axis.set_axisbelow(True)
        axis.grid(axis="y", alpha=0.2)
        axis.set_ylim(bottom=0)
        if title == "Task success rate":
            axis.set_ylim(0, 1.12)
        for position, value in zip(positions, values):
            axis.annotate("N/A" if value is None else f"{value:,.2f}",
                          (position, 0 if value is None else value), xytext=(0, 4),
                          textcoords="offset points", ha="center", fontsize=9)
    FigureCanvasSVG(figure).print_svg(directory / "comparison.svg")
    FigureCanvasAgg(figure).print_png(directory / "comparison.png")
    svg = (directory / "comparison.svg").read_text(encoding="utf-8")
    return svg[svg.index("<svg"):]


def _html(report: ComparisonReport, chart: str) -> str:
    rows = []
    for row in report.rows:
        aggregate = row.aggregate
        success = "未知" if aggregate.task_success_rate is None else f"{aggregate.task_success_rate:.1%}"
        rows.append(
            "<tr>" + "".join(f"<td>{html.escape(str(value))}</td>" for value in [
                row.name, ", ".join(row.models), f"{aggregate.run_count}/{row.planned_runs}", success,
                aggregate.total_tokens if aggregate.total_tokens is not None else "未知",
                f"{aggregate.average_elapsed_seconds:.2f}" if aggregate.average_elapsed_seconds is not None else "未知",
                f"{aggregate.tool_metrics_runs}/{aggregate.run_count}",
                f"{aggregate.token_complete_runs}/{aggregate.run_count}",
                f"{row.paired_wins}/{row.paired_losses}/{row.paired_ties}",
            ]) + "</tr>"
        )
    failures = []
    for run in report.runs:
        if run["status"] != "passed":
            detail = run["error"] or "未提供详情"
            if run["input_limit"] is not None:
                detail += f"。输入估算 {run['estimated_input']}，上限 {run['input_limit']}"
            values = [run[key] for key in ["variant", "repeat", "task_id", "status", "failure_type"]]
            values.append(detail)
            failures.append("<tr>" + "".join(f"<td>{html.escape(str(value))}</td>" for value in values) + "</tr>")
    conditions = "\n".join(
        f"<h3>{html.escape(row.name)}</h3><p>服务标识 {html.escape(', '.join(row.provider_ids))}</p>"
        f"<pre>{html.escape(row.options.model_dump_json(indent=2))}</pre>"
        for row in report.rows
    )
    tool_rows = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(value))}</td>" for value in [
            row.name, row.aggregate.tools.total.attempted, row.aggregate.tools.total.succeeded,
            row.aggregate.tools.total.failed, row.aggregate.tools.total.interrupted,
            f"{row.aggregate.tool_success_rate:.1%}" if row.aggregate.tool_success_rate is not None else "未知",
        ]) + "</tr>" for row in report.rows
    )
    findings = []
    for row in report.rows[1:]:
        token_delta = "未知" if row.paired_mean_token_delta is None else f"{row.paired_mean_token_delta:+,.1f} Token"
        latency_delta = "未知" if row.paired_mean_latency_delta is None else f"{row.paired_mean_latency_delta:+.2f} 秒"
        findings.append(
            f"<p>{html.escape(row.name)} 与基线有 {row.matched_runs} 组配对。"
            f"成功新增 {row.paired_wins}，减少 {row.paired_losses}，结果相同 {row.paired_ties}。"
            f"配对平均每题用量差为 {token_delta}，耗时差为 {latency_delta}。</p>"
        )
    return f'''<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(report.name)} 评测报告</title>
<style>body{{font:16px/1.6 system-ui,sans-serif;color:#172033;max-width:1180px;margin:40px auto;padding:0 24px}}
table{{border-collapse:collapse;width:100%;margin:20px 0}}th,td{{padding:10px;border-bottom:1px solid #dce1ea;text-align:left}}
th{{background:#f3f5fa}}pre{{background:#f3f5fa;padding:16px;overflow:auto}}svg{{width:100%;height:auto}}code{{overflow-wrap:anywhere}}</style>
<h1>{html.escape(report.name)} 评测报告</h1>
<p>状态为 {html.escape(report.state)}。任务 {len(report.task_ids)} 个，每个条件重复 {report.repeats} 次。
基线为 {html.escape(report.baseline)}，配对比较按相同任务与重复序号进行。</p>
<p>任务指纹 <code>{report.suite_sha256}</code></p>
<p>Docker 镜像 <code>{html.escape(report.image_id or '未知')}</code></p>
<p>成功率以已记录运行次数为分母，包含环境错误与取消。覆盖率展示已记录与计划次数。
工具覆盖与用量覆盖分别展示可计量次数。未知总量显示为未知或 N/A。</p>
<table><thead><tr><th>条件</th><th>模型</th><th>运行/计划</th><th>成功率</th><th>Token</th><th>平均秒</th>
<th>工具覆盖</th><th>用量覆盖</th><th>配对胜/负/平</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
{chart}
<h2>配对结果</h2>{''.join(findings)}
<h2>工具调用</h2><table><thead><tr><th>条件</th><th>尝试</th><th>成功</th><th>失败</th><th>中断</th><th>成功率</th></tr></thead>
<tbody>{tool_rows}</tbody></table>
<h2>失败记录</h2><table><thead><tr><th>条件</th><th>重复</th><th>任务</th><th>状态</th><th>类型</th><th>详情</th></tr></thead>
<tbody>{''.join(failures) if failures else '<tr><td colspan="6">本次记录全部通过</td></tr>'}</tbody></table>
<h2>有效配置</h2>{conditions}
<p>上下文预算沿用 UTF-8 字节估算。图表统计来自本任务集与重复次数，时间包含环境执行和清理。
各条件按计划顺序串行运行。完整结果与覆盖数据见 report.json，逐题记录见 runs.csv。</p></html>'''


def write_report(experiment_directory: str | Path, output_directory: str | Path | None = None) -> Path:
    report = compare_experiment(experiment_directory)
    root = Path(experiment_directory).resolve(strict=True)
    destination = root / f"report-{uuid.uuid4().hex[:8]}" if output_directory is None else Path(output_directory).resolve()
    if destination.exists():
        raise ValueError("报告目录已存在，请使用新路径")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".report-", dir=destination.parent) as temporary:
        staging = Path(temporary) / "report"
        staging.mkdir()
        chart = _chart(report, staging)
        _save_json(staging / "report.json", report)
        with (staging / "runs.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            fields = ["variant", "repeat", "task_id", "model", "status", "failure_type", "elapsed_seconds", "total_tokens", "tool_calls", "error", "estimated_input", "input_limit", "result_file"]
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(report.runs)
        (staging / "report.html").write_text(_html(report, chart), encoding="utf-8")
        staging.rename(destination)
    return destination
