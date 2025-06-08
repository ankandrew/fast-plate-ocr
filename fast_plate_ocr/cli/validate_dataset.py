"""
Validate a `fast-plate-ocr` dataset before training.
"""

import sys
from collections import Counter
from pathlib import Path

import click
import pandas as pd
from PIL import Image, UnidentifiedImageError
from rich import box
from rich.console import Console
from rich.markup import escape
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskID, TextColumn
from rich.table import Table

from fast_plate_ocr.train.model.config import load_plate_config_from_yaml

console = Console()


def partial_decode_ok(path: Path) -> tuple[bool, tuple[int, int] | None]:
    """
    Return (is_ok, (h, w)) – True only if Pillow can verify the file and
    obtain its dimensions without allocating the full pixel buffer.
    """
    try:
        with Image.open(path) as im:
            im.verify()  # light-weight integrity check
            w, h = im.size
            return True, (h, w)
    except (UnidentifiedImageError, OSError):
        return False, None


def validate_dataset(
    df: pd.DataFrame,
    cfg,
    min_h: int,
    min_w: int,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]], pd.DataFrame]:
    """
    Iterate over the dataframe, collect errors and warnings, and return a cleaned df.
    """
    errors, warnings, ok_rows = [], [], []
    char_counter: Counter[str] = Counter()
    seen_paths: set[Path] = set()

    # pretty progress bar
    progress = Progress(
        SpinnerColumn(),
        BarColumn(bar_width=None),
        TextColumn("{task.completed}/{task.total}"),
        TextColumn("[bold blue]{task.description}"),
        transient=True,
        console=console,
    )
    task: TaskID = progress.add_task("Scanning images", total=len(df))

    with progress:
        for row_idx, row in enumerate(df.itertuples(index=False)):
            img_path = Path(str(row.image_path))
            plate = str(row.plate_text)
            char_counter.update(plate)
            line_no = str(row_idx + 2)

            # Check file exists
            if not img_path.exists():
                errors.append((line_no, f"Missing image file: {img_path}"))
                progress.update(task, advance=1)
                continue

            # Check decodable & size
            ok, shape = partial_decode_ok(img_path)
            if not ok or shape is None:
                errors.append((line_no, f"Corrupt or unreadable image: {img_path}"))
                progress.update(task, advance=1)
                continue

            h, w = shape
            if h < min_h or w < min_w:
                warnings.append((line_no, f"Tiny image ({h}x{w} < {min_h}x{min_w}): {img_path}"))

            # Check resize feasibility (>= 1 px in each axis)
            r = min(cfg.img_height / h, cfg.img_width / w)
            new_w, new_h = round(w * r), round(h * r)
            if new_w == 0 or new_h == 0:
                errors.append((line_no, f"Resize would give 0x0 ({new_h}x{new_w}) from {img_path}"))
                progress.update(task, advance=1)
                continue

            # Check plate text length and alphabet
            if len(plate) > cfg.max_plate_slots:
                errors.append(
                    (
                        line_no,
                        f"Plate too long ({len(plate)}>{cfg.max_plate_slots}):"
                        f" '{plate}' [{img_path}]",
                    )
                )
                progress.update(task, advance=1)
                continue

            bad_chars = set(plate) - set(cfg.alphabet)
            if bad_chars:
                errors.append(
                    (line_no, f"Invalid chars {bad_chars} in plate '{plate}' [{img_path}]")
                )
                progress.update(task, advance=1)
                continue

            # Check duplicate paths
            if img_path in seen_paths:
                warnings.append((line_no, f"Duplicate image entry: {img_path}"))
                progress.update(task, advance=1)
                continue
            seen_paths.add(img_path)

            ok_rows.append(row)
            progress.update(task, advance=1)

    alphabet_no_pad = cfg.alphabet.replace(cfg.pad_char, "")
    unused_chars = sorted(set(alphabet_no_pad) - set(char_counter))
    if unused_chars:
        warnings.append(("-", f"Character(s) not found in any plate: {', '.join(unused_chars)}"))
    cleaned_df = pd.DataFrame(ok_rows)
    return errors, warnings, cleaned_df


def rich_report(errors, warnings):
    summary = Table(title="Validation Summary", box=box.SQUARE, expand=False)
    summary.add_column("Category", style="bold")
    summary.add_column("Count", justify="right")
    summary.add_row("Errors", str(len(errors)), style="red" if errors else "green")
    summary.add_row("Warnings", str(len(warnings)), style="yellow" if warnings else "green")
    console.print()
    console.print(summary)

    def dump(name, rows, style):
        if not rows:
            return
        tbl = Table(title=name, box=box.SQUARE, header_style=style, show_lines=False, expand=False)
        tbl.add_column("Line", justify="right", style=style)
        tbl.add_column("Message", style=style, overflow="fold")

        for line_no, msg in rows:
            tbl.add_row(str(line_no), escape(msg))
        console.print()
        console.print(tbl)

    dump("Errors", errors, "red")
    dump("Warnings", warnings, "yellow")


@click.command(context_settings={"max_content_width": 120})
@click.option(
    "--annotations-file",
    "-a",
    required=True,
    type=click.Path(exists=True, dir_okay=False, file_okay=True, path_type=Path),
    help="CSV with image_path and plate_text columns.",
)
@click.option(
    "--plate-config-file",
    required=True,
    type=click.Path(exists=True, dir_okay=False, file_okay=True, path_type=Path),
    help="Path to the OCR YAML config with image dimensions & alphabet.",
)
@click.option(
    "--warn-only",
    is_flag=True,
    help="Exit 0 even if errors occur.",
)
@click.option(
    "--export-fixed",
    type=str,
    help="Filename for the cleaned CSV written in the same directory as the annotations file.",
)
@click.option(
    "--min-height",
    default=2,
    show_default=True,
    type=int,
    help="Minimum allowed image height.",
)
@click.option(
    "--min-width",
    default=2,
    show_default=True,
    type=int,
    help="Minimum allowed image width.",
)
def main(
    annotations_file: Path,
    plate_config_file: Path,
    warn_only: bool,
    export_fixed: str | None,
    min_height: int,
    min_width: int,
):
    """
    Script to validate the dataset before training.
    """
    cfg = load_plate_config_from_yaml(plate_config_file)

    df = pd.read_csv(annotations_file)
    csv_root = annotations_file.parent
    df["image_path"] = df["image_path"].apply(lambda p: str((csv_root / p).resolve()))

    errors, warnings, cleaned = validate_dataset(df, cfg, min_height, min_width)

    # Make cleaned dataset img_path relative (expected format)
    cleaned["image_path"] = cleaned["image_path"].apply(
        lambda p: str(Path(p).relative_to(csv_root))
    )

    rich_report(errors, warnings)

    if export_fixed:
        export_path = csv_root / Path(export_fixed).name
        if export_path.resolve() == annotations_file.resolve():
            console.print(
                "[yellow]⚠️ Skipping export: make sure you don't "
                "overwrite original annotations file.[/]"
            )
        elif export_path.exists():
            console.print(f"[yellow]⚠️ Skipping export: file already exists at {export_path}[/]")
        else:
            cleaned.to_csv(export_path, index=False)
            console.print(
                f"[green]✅ Wrote cleaned CSV with {len(cleaned)} rows at {export_path} [/]"
            )

    if errors and not warn_only:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
