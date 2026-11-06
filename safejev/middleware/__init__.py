"""Middleware: tool decorators (``SafeJEV.guard_tool``) and the MCP screening proxy."""
from .mcp_proxy import MCPProxy, wrap_mcp_config

__all__ = ["MCPProxy", "wrap_mcp_config"]
