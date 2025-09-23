from typing import Any, Dict, List, Optional, Union

import ray
from langchain.tools import BaseTool


@ray.remote(concurrency_groups={"io": 1, "compute": 5})
class ToolWorker:
    """
    A unified wrapper class for LangChain tools, enabling a standard
    interface to call tools regardless of their internal differences.
    """

    def __init__(self, tools: List[BaseTool]):
        """
        Initialize ToolWorker with a list of LangChain tools.

        Args:
            tools (List[BaseTool]): List of LangChain tools to register.
        """
        self._tool_registry: Dict[str, BaseTool] = {tool.name: tool for tool in tools}

    @ray.method(concurrency_group="io")
    def list_tools(self) -> List[str]:
        """Return the list of available tool names."""
        return list(self._tool_registry.keys())

    @ray.method(concurrency_group="io")
    def get_tool_description(self, tool_name: str) -> Optional[str]:
        """Return the description of a specific tool."""
        tool = self._tool_registry.get(tool_name)
        return tool.description if tool else None

    @ray.method(concurrency_group="io")
    def get_args_schema(self, tool_name: str):
        """Return the argument schema of a specific tool."""
        assert tool_name in self._tool_registry, f"Tool '{tool_name}' not found. Available: {self.list_tools()}"
        tool = self._tool_registry.get(tool_name)
        schema = tool.args_schema.model_json_schema(by_alias=False)
        return schema

    @ray.method(concurrency_group="compute")
    def call(self, tool_name: str, input_data: Union[str, Dict[str, Any]], **kwargs) -> Any:
        """
        Call a tool by name with input data.

        Args:
            tool_name (str): Name of the tool to call.
            input_data (Union[str, Dict[str, Any]]): Input to pass to the tool.
            **kwargs: Extra keyword arguments for the tool.

        Returns:
            Any: The tool's output.
        """
        if tool_name == "return_parsing_error":
            return "Error: Tool call parsing error. Please use the correct JSON format."
        if tool_name not in self._tool_registry:
            return f"Error: Tool {tool_name} not found. Available tools: {self.list_tools()}"
        tool = self._tool_registry[tool_name]
        try:
            ret = tool.run(input_data, **kwargs)
        except Exception as e:
            ret = f"Error: Tool {tool_name} execution failed with error: {str(e)}"
        return ret
