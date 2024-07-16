"""
Utils used for the CLI scripts.
"""

import inspect
import pathlib
from collections.abc import Callable
from functools import wraps
from typing import Any

from rich import box
from rich.console import Console
from rich.table import Table


def color_for_dtype(value: Any) -> str:
    """
    Returns a color name based on the data type of the input value.

    Args:
        value (Any): The input value whose type is to be determined.

    Returns:
        str: The color name as a string.
    """
    if isinstance(value, bool):
        return "green" if value else "red"
    if isinstance(value, int):
        return "cyan"
    if isinstance(value, float):
        return "magenta"
    if isinstance(value, pathlib.Path):
        return "yellow"
    if value is None:
        return "grey50"
    return "white"


def print_variables_as_table(
    c1_title: str, c2_title: str, title: str = "Variables Table", **kwargs: Any
) -> None:
    """
    Prints variables in a formatted table using the rich library.

    Args:
        c1_title (str): Title of the first column.
        c2_title (str): Title of the second column.
        title (str): Title of the table.
        **kwargs (Any): Variable names and values to be printed.
    """
    console = Console()
    table = Table(title=title, show_header=True, header_style="bold magenta", box=box.ROUNDED)
    table.add_column(c1_title, min_width=20, justify="left", style="bold")
    table.add_column(c2_title, min_width=60, justify="left", style="bold")
    for key, value in kwargs.items():
        color = color_for_dtype(value)
        table.add_row(f"[bold]{key}[/bold]", f"[{color}]{value}[/{color}]")
    console.print(table)


def print_params(
    table_title: str = "Parameters Table", c1_title: str = "Variable", c2_title: str = "Value"
) -> Callable:
    """
    A decorator that prints the parameters of a function in a formatted table
    using the rich library.

    Args:
        c1_title (str, optional): Title of the first column. Defaults to "Variable".
        c2_title (str, optional): Title of the second column. Defaults to "Value".
        table_title (str, optional): Title of the table. Defaults to "Parameters Table".

    Returns:
        Callable: The wrapped function with parameter printing functionality.
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            func_signature = inspect.signature(func)
            bound_arguments = func_signature.bind(*args, **kwargs)
            bound_arguments.apply_defaults()
            params = dict(bound_arguments.arguments.items())
            print_variables_as_table(c1_title, c2_title, table_title, **params)
            return func(*args, **kwargs)

        return wrapper

    return decorator
