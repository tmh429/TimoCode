"""组装 Phase 2 的基础 Coding Tools。"""

from timocode.security.permissions import PermissionManager
from timocode.security.workspace import Workspace
from timocode.tools.base import Tool
from timocode.tools.filesystem import EditFileTool, ReadFileTool, WriteFileTool
from timocode.tools.search import ListFilesTool, SearchFilesTool
from timocode.tools.shell import ShellTool


def create_coding_tools(workspace: Workspace, permissions: PermissionManager) -> list[Tool]:
    return [
        ReadFileTool(workspace), WriteFileTool(workspace, permissions),
        EditFileTool(workspace, permissions), ListFilesTool(workspace),
        SearchFilesTool(workspace), ShellTool(workspace, permissions),
    ]
