"""副作用操作必须由调用方明确批准。"""

import asyncio
import json
import os
import select
import sys
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TextIO


@dataclass(frozen=True)
class PermissionRequest:
    tool: str
    details: dict[str, str | int]


ApprovalCallback = Callable[[PermissionRequest], Awaitable[bool]]


class PermissionManager:
    def __init__(self, confirm: ApprovalCallback | None = None):
        self._confirm = confirm

    async def require(self, request: PermissionRequest) -> None:
        if self._confirm is None or await self._confirm(request) is not True:
            raise PermissionError(f"用户未批准 {request.tool} 操作")


async def confirm_console(request: PermissionRequest, *, output: TextIO | None = None) -> bool:
    output = sys.stdout if output is None else output
    if not sys.stdin.isatty():
        return False
    print(f"\n[permission] {request.tool}", file=output, flush=True)
    print(json.dumps(request.details, ensure_ascii=False, indent=2), file=output, flush=True)
    print("输入 yes 批准，其他输入拒绝 > ", file=output, end="", flush=True)
    if os.name != "nt":
        while not select.select([sys.stdin], [], [], 0)[0]:
            await asyncio.sleep(0.05)
        return sys.stdin.readline().strip().lower() == "yes"

    import msvcrt

    answer = ""
    while True:
        if not msvcrt.kbhit():
            await asyncio.sleep(0.05)
            continue
        char = msvcrt.getwch()
        if char == "\x03":
            raise asyncio.CancelledError
        if char in {"\x00", "\xe0"}:
            msvcrt.getwch()
        elif char in {"\r", "\n"}:
            print(file=output, flush=True)
            return answer.strip().lower() == "yes"
        elif char == "\b":
            if answer:
                answer = answer[:-1]
                print("\b \b", file=output, end="", flush=True)
        elif char.isprintable() and len(answer) < 32:
            answer += char
            print(char, file=output, end="", flush=True)
