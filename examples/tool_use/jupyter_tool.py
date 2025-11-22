"""
Tool sandbox module for safe code execution and tool management.

This module provides:
- PythonSandbox: Safe Python code execution environment
- ToolRegistry: Tool registration and execution management
- Memory management utilities
"""

import asyncio
import httpx
import os
import requests
import traceback
from typing import Any, Dict, List

# Configuration for tool execution
TOOL_CONFIGS = {
    "tool_concurrency": 360,
    # Python interpreter settings
    "python_timeout": 120,  # 2 minutes for complex calculations
}

# Global semaphore for controlling concurrent tool executions
SEMAPHORE = asyncio.Semaphore(TOOL_CONFIGS["tool_concurrency"])


class JupyterToolClient:
    """Client for interacting with the Jupyter tool service"""
    def __init__(self, server_url: str, http_timeout: int = 300):
        self.http_client = httpx.AsyncClient(timeout=httpx.Timeout(http_timeout))

        self.default_server_url = "http://localhost:8000"
        self.server_url = server_url or os.getenv("JUPYTER_TOOL_SERVER_URL", self.default_server_url)
        print(f"JupyterToolClient initialized with server_url: {self.server_url}")

    async def execute_code(self, session_id: str, code: str) -> str:
        """Execute Python code in the Jupyter tool service"""
        url = f"{self.server_url}/execute_code"
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "X-Session-ID": session_id,
        }

        payload = {"session_id": session_id, "code": code}

        try:
            response = await self.http_client.post(url, json=payload or {}, headers=headers)
            if response.status_code == 200:
                return response.json().get("output", "")
            else:
                err = f"Exception occurred when calling Jupyter notebook server. HTTP response code: {response.status_code}, HTTP response: {response.text}"
                print((f"[session_id: {session_id}] {err}"))
                return err
        except Exception as e:
            err = f"Exception occurred when calling Jupyter notebook server: {e}"
            print((f"[session_id: {session_id}] {err}\ntraceback: {traceback.format_exc()}"))
            return err

    async def end_session(self, session_id: str) -> str:
        """End the Jupyter tool session"""
        url = f"{self.server_url}/end_session/{session_id}"
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "X-Session-ID": session_id,
        }

        try:
            response = await self.http_client.post(url, headers=headers)
            if response.status_code == 200:
                return response.json().get("output", "")
            else:
                err = f"Failed to end session {session_id}. HTTP response code: {response.status_code}, HTTP response: {response.text}"
                print((f"[session_id: {session_id}] {err}"))
                return err
        except Exception as e:
            err = f"Exception occurred when ending Jupyter notebook server session: {e}"
            print((f"[session_id: {session_id}] {err}\ntraceback: {traceback.format_exc()}"))
            return err


class ToolRegistry:
    """Tool registry, manages available tools and their execution"""

    def __init__(self):
        self.tools = {}
        self._register_default_tools()
        self.jupyter_client = JupyterToolClient(server_url=None)

    def _register_default_tools(self):
        """Register default tools in the registry"""
        # Python code interpreter
        self.register_tool(
            "python",
            {
                "type": "function",
                "function": {
                    "name": "python",
                    "description": "Use this tool to execute Python code in your chain of thought. The code will not be shown to the user. This tool should be used for internal reasoning, but not for code that is intended to be visible to the user (e.g. when creating plots, tables, or files).\nWhen you send a message containing Python code to python, it will be executed in a stateful Jupyter notebook environment. python will respond with the output of the execution or time out after 120.0 seconds. Internet access for this session is UNAVAILABLE.",
                    "parameters": {
                        "type": "object",
                        "properties": {"code": {"type": "string", "description": "The Python code to execute."}},
                        "required": ["code"],
                    },
                },
            },
        )

    def register_tool(self, name: str, tool_spec: Dict[str, Any]):
        """Register a new tool in the registry"""
        self.tools[name] = tool_spec

    def get_tool_specs(self) -> List[Dict[str, Any]]:
        """Get all tool specifications as a list"""
        return list(self.tools.values())

    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any], session_id: str) -> str:
        """Execute a tool call with the given arguments"""
        if tool_name not in self.tools:
            return f"Error: Tool '{tool_name}' not found"

        if tool_name == "python":
            return await self._execute_python(arguments, session_id)
        else:
            return f"Error: Tool '{tool_name}' not implemented"

    async def _execute_python(self, arguments: Dict[str, Any], session_id: str) -> str:
        """Execute Python code using the sandbox"""
        code = arguments.get("code", "")
        if not code.strip():
            return "Error: No code provided"

        # Request jupyter tool service
        result = await self.jupyter_client.execute_code(session_id, code)
        return result

# Global tool registry instance
tool_registry = ToolRegistry()


async def _test():
    """Test the tool registry and Jupyter tool client"""
    session_id = "test_session"
    code = "import math\na=math.sqrt(16)\na"
    result = await tool_registry.execute_tool("python", {"code": code}, session_id)
    print(f"Execution result: {result}")
    
    code = "a = a + 10\na"
    result = await tool_registry.execute_tool("python", {"code": code}, session_id)
    print(f"Execution result: {result}")
    
    code = "b = a * 5\na, b"
    result = await tool_registry.execute_tool("python", {"code": code}, session_id)
    print(f"Execution result: {result}")

    code = "import time\ntime.sleep(130)\n"
    result = await tool_registry.execute_tool("python", {"code": code}, session_id)
    print(f"Execution result: {result}")

    result = await tool_registry.jupyter_client.end_session(session_id)
    print(f"End session result: {result}")


if __name__ == "__main__":
   asyncio.run(_test())