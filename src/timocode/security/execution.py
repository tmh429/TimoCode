"""有界子进程输出、超时与进程树回收，供命令后端共用。"""

import asyncio
import os
import signal
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TypeVar

from timocode.security.processes import WindowsJob


T = TypeVar("T")


async def complete_task(task: asyncio.Task[T]) -> T:
    """等待资源操作收尾，优先报告失败，其次传播取消。"""
    cancelled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled = True
        except Exception:
            break
    result = task.result()
    if cancelled:
        raise asyncio.CancelledError
    return result


@dataclass(frozen=True)
class ProcessResult:
    exit_code: int
    stdout: str
    stderr: str
    stdout_truncated: bool = False
    stderr_truncated: bool = False
    timed_out: bool = False

    def as_dict(self) -> dict:
        return asdict(self)


async def _capture(stream: asyncio.StreamReader, limit: int) -> tuple[str, bool]:
    kept = bytearray()
    size = 0
    while chunk := await stream.read(65536):
        size += len(chunk)
        kept.extend(chunk[:max(0, limit - len(kept))])
    return kept.decode("utf-8", errors="replace"), size > limit


async def run_process(
    command: list[str], *, cwd: Path, timeout_seconds: float,
    max_output_bytes: int, env: dict[str, str] | None = None,
) -> ProcessResult:
    job = WindowsJob() if os.name == "nt" else None
    process = None
    readers = []
    timed_out = False
    try:
        spawning = asyncio.create_task(asyncio.create_subprocess_exec(
            *command, cwd=cwd, stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            start_new_session=os.name != "nt",
            env=env,
            creationflags=0x4 if os.name == "nt" else 0,
        ))
        cancelled = False
        while not spawning.done():
            try:
                await asyncio.shield(spawning)
            except asyncio.CancelledError:
                cancelled = True
        process = spawning.result()
        readers = [
            asyncio.create_task(_capture(process.stdout, max_output_bytes)),
            asyncio.create_task(_capture(process.stderr, max_output_bytes)),
        ]
        if cancelled:
            raise asyncio.CancelledError
        if job is not None:
            job.attach(process.pid)
        try:
            async with asyncio.timeout(timeout_seconds):
                # shield 保留输出读取任务，超时后仍可收集已截断的输出。
                await asyncio.shield(asyncio.gather(*readers))
                await process.wait()
        except TimeoutError:
            timed_out = True
            await _terminate(process, job)
        stdout, stderr = await asyncio.gather(*readers)
        return ProcessResult(
            exit_code=process.returncode, stdout=stdout[0], stderr=stderr[0],
            stdout_truncated=stdout[1], stderr_truncated=stderr[1], timed_out=timed_out,
        )
    finally:
        try:
            if process is not None:
                await complete_task(asyncio.create_task(_terminate(process, job)))
        finally:
            if job is not None:
                job.close()
            if readers:
                await asyncio.gather(*readers, return_exceptions=True)


async def _terminate(process: asyncio.subprocess.Process, job: WindowsJob | None) -> None:
    if job is not None:
        job.terminate()
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    if process.returncode is None:
        try:
            process.kill()
        except ProcessLookupError:
            pass
    await process.wait()
