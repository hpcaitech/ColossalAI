from langchain_core.tools import Tool
from langchain_experimental.utilities import PythonREPL
from pydantic import BaseModel, Field
from pydantic.fields import FieldInfo


def make_title(field_name: str, field_info: FieldInfo) -> str:
    return field_name


class PythonInput(BaseModel):
    code: str = Field(description="The python code to execute", field_title_generator=make_title)


python_repl = PythonREPL()


def run_python_code(code: str) -> str:
    if code.startswith("```python"):
        code = code.replace("```python", "```", 1).strip()
    if code.startswith("```py"):  # qwen3 uses ```py
        code = code.replace("```py", "```", 1).strip()
    return python_repl.run(code, timeout=20)


repl_tool = Tool(
    name="python_repl",
    description="A Python shell. Use this to execute python commands. Input should be a valid python command. If you want to see the output of a value, you should print it out with `print(...)`.",
    func=run_python_code,
    args_schema=PythonInput,
)
