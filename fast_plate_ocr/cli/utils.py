"""
Utils used for the CLI scripts.
"""

import importlib.util
import inspect
import pathlib
import random
from collections.abc import Callable, Sequence
from functools import wraps
from typing import Any

import albumentations as A
import numpy as np
from rich import box
from rich.console import Console
from rich.pretty import Pretty
from rich.table import Table


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
    console.print("\n")
    table = Table(title=title, show_header=True, header_style="bold blue", box=box.ROUNDED)
    table.add_column(c1_title, min_width=20, justify="left", style="bold")
    table.add_column(c2_title, min_width=60, justify="left", style="bold")

    for key, value in kwargs.items():
        if isinstance(value, pathlib.Path):
            value = str(value)  # noqa: PLW2901
        table.add_row(f"[bold]{key}[/bold]", Pretty(value))

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


def print_train_details(augmentation: A.Compose, config: dict[str, Any]) -> None:
    console = Console()
    console.print("\n")
    console.print("[bold blue]Augmentation Pipeline:[/bold blue]")
    console.print(Pretty(augmentation))
    console.print("\n")
    console.print("[bold blue]Configuration:[/bold blue]")
    console.print(Pretty(config))
    console.print("\n")


def requires(*modules: str, pkg_name: Sequence[str] | None = None) -> Callable:
    """
    Decorator that checks if given modules are importable. If not, raises ModuleNotFoundError with
    a hint to install the package(s).

    Args:
        modules (str): Names of modules to check (via importlib.util.find_spec).
        pkg_name (Sequence[str] | None): Names of packages to suggest installing.

    Returns:
        Callable: The wrapped function that checks for module availability.
    """

    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any):
            missing = [m for m in modules if importlib.util.find_spec(m) is None]
            if missing:
                pkg_missing = " ".join(pkg_name or missing)
                raise ModuleNotFoundError(
                    f"Cannot run `{fn.__name__}` because {missing!r} "
                    f"is not installed. Please install the required package(s): {pkg_missing}"
                )
            return fn(*args, **kwargs)

        return wrapper

    return decorator


def seed_everything(seed: int) -> None:
    """
    Seed random number generators for reproducibility.

    Args:
        seed (int): The seed value to set.
    """
    random.seed(seed)
    np.random.seed(seed)
