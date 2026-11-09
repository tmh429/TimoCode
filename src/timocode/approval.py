"""终端内使用方向键选择的逐次操作审批。"""

import asyncio
import sys
from contextlib import closing
from typing import TextIO

from prompt_toolkit.application import Application
from prompt_toolkit.input import create_input
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Layout, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.output import create_output
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.panel import Panel

from timocode.security.permissions import PermissionRequest
from timocode.terminal import literal_text


async def confirm_console(request: PermissionRequest, *, output: TextIO | None = None) -> bool:
    if not sys.stdin.isatty():
        return False
    output = sys.stderr if output is None else output
    console = Console(file=output)
    details = request.details
    lines = []
    for key, label in (("path", "文件"), ("command", "命令"), ("cwd", "目录"), ("image", "镜像")):
        if key in details:
            lines.append(f"{label}  {details[key]}")
    if "content_preview" in details:
        lines.extend(["", "修改预览", str(details["content_preview"])])
        if details.get("preview_truncated"):
            lines.append("（预览已截断）")
    console.print(Panel(literal_text("\n".join(lines)), title="需要批准本次操作", border_style="yellow"))
    selected = 0
    bindings = KeyBindings()

    @bindings.add("up")
    @bindings.add("down")
    def move(event):
        nonlocal selected
        selected = 1 - selected
        event.app.invalidate()

    @bindings.add("enter")
    def accept(event):
        event.app.exit(result=selected == 0)

    @bindings.add("escape")
    @bindings.add("c-d")
    @bindings.add("c-z")
    def reject(event):
        event.app.exit(result=False)

    @bindings.add("c-c")
    def cancel(event):
        event.app.exit(exception=asyncio.CancelledError())

    def menu():
        fragments = []
        for index, label in enumerate(("允许本次", "拒绝")):
            prefix = "> " if selected == index else "  "
            style = "class:selected" if selected == index else ""
            fragments.append((style, prefix + label + "\n"))
        fragments.append(("", "↑/↓ 选择 · Enter 确认 · Esc 拒绝"))
        return fragments

    with closing(create_input(stdin=sys.stdin)) as terminal_input:
        application = Application(
            layout=Layout(Window(FormattedTextControl(menu), height=3)),
            key_bindings=bindings, input=terminal_input, output=create_output(stdout=output),
            style=Style.from_dict({"selected": "bold ansicyan"}), erase_when_done=True,
        )
        try:
            allowed = await application.run_async()
        except EOFError:
            allowed = False
    console.print("已批准本次操作" if allowed else "已拒绝本次操作", style="dim")
    return allowed
