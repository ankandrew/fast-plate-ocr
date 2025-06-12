"""
Display statistics for a `fast-plate-ocr` dataset.
"""

from collections import Counter
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import click
import pandas as pd
from PIL import Image, UnidentifiedImageError
from rich import box
from rich.console import Console, Group
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from fast_plate_ocr.train.model.config import load_plate_config_from_yaml

# pylint: disable=too-many-locals

console = Console()


def _header_shape(path: Path) -> tuple[bool, tuple[int, int] | None]:
    try:
        with Image.open(path) as im:
            im.verify()
            w, h = im.size
            return True, (h, w)
    except (UnidentifiedImageError, OSError):
        return False, None


def _compact_table(title: str, values: Sequence[float]) -> Table:
    s = pd.Series(values, dtype="float64")
    desc = s.describe(percentiles=[0.05, 0.5, 0.95])
    metrics = ["count", "mean", "std", "min", "max", "5%", "50%", "95%"]
    tbl = Table(title=title, box=box.MINIMAL_DOUBLE_HEAD, pad_edge=False, expand=False)
    for m in metrics:
        tbl.add_column(m, justify="right", style="bold")
    tbl.add_row(*[f"{desc[m]:.2f}" if pd.notna(desc[m]) else "-" for m in metrics])
    return tbl


@click.command(context_settings={"max_content_width": 120})
@click.option(
    "--annotations",
    "-a",
    required=True,
    type=click.Path(exists=True, dir_okay=False, file_okay=True, path_type=Path),
    help="CSV with image_path and plate_text columns.",
)
@click.option(
    "--plate-config-file",
    "-c",
    required=True,
    type=click.Path(exists=True, dir_okay=False, file_okay=True, path_type=Path),
    help="YAML config so we know alphabet/pad char.",
)
@click.option(
    "--top-chars",
    default=10,
    show_default=True,
    type=int,
    help="Show N most frequent characters.",
)
@click.option(
    "--workers",
    default=8,
    show_default=True,
    type=int,
    help="Parallel header reads (0 disables threading).",
)
def main(annotations: Path, plate_config_file: Path, top_chars: int, workers: int) -> None:
    """
    Display statistics for a `fast-plate-ocr` dataset.
    """
    plate_config = load_plate_config_from_yaml(plate_config_file)

    df_annots = pd.read_csv(annotations)
    root = annotations.parent
    df_annots["image_path"] = df_annots["image_path"].apply(lambda p: str((root / p).resolve()))

    # Plate lengths and char frequencies
    plate_lengths = df_annots["plate_text"].str.len().tolist()
    char_counter: Counter[str] = Counter("".join(df_annots["plate_text"].tolist()))

    # File extension counts
    ext_counter = Counter(df_annots["image_path"].apply(lambda p: Path(p).suffix.lower()))

    # Image header dimensions
    paths = [Path(p) for p in df_annots["image_path"].tolist()]
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            dims = list(ex.map(_header_shape, paths))
    else:
        dims = [_header_shape(p) for p in paths]

    valid_dims = [dims_pair for ok, dims_pair in dims if ok and dims_pair is not None]

    heights = [h for h, _ in valid_dims]
    widths = [w for _, w in valid_dims]
    aspects = [w / h for h, w in valid_dims if h > 0]

    # Build tables
    tbl_len = _compact_table("Plate Lengths", plate_lengths)
    tbl_h = _compact_table("Image Height", heights)
    tbl_w = _compact_table("Image Width", widths)
    tbl_ar = _compact_table("Aspect Ratio", aspects)

    # Extension table
    tbl_ext = Table(title="Extensions", box=box.MINIMAL_DOUBLE_HEAD, pad_edge=False)
    tbl_ext.add_column("Ext", style="bold", justify="left")
    tbl_ext.add_column("Count", justify="right")
    for ext, cnt in ext_counter.most_common():
        tbl_ext.add_row(ext or "<none>", str(cnt))

    # Character freq table
    tbl_char = Table(title=f"Top {top_chars} Chars", box=box.MINIMAL_DOUBLE_HEAD, pad_edge=False)
    tbl_char.add_column("Char", style="bold")
    tbl_char.add_column("Count", justify="right")
    for ch, cnt in char_counter.most_common(top_chars):
        if ch == plate_config.pad_char:
            continue
        tbl_char.add_row(escape(ch), str(cnt))

    group = Group(tbl_len, tbl_h, tbl_w, tbl_ar, tbl_ext, tbl_char)
    console.print(
        Panel.fit(group, title="Dataset Statistics", border_style="green", box=box.SQUARE)
    )


if __name__ == "__main__":
    main()
