"""
Main CLI used when training a FastPlateOCR model.
"""

try:
    import click

    from fast_plate_ocr.cli.onnx_converter import export_onnx
    from fast_plate_ocr.cli.train import train
    from fast_plate_ocr.cli.valid import valid
    from fast_plate_ocr.cli.visualize_augmentation import visualize_augmentation
    from fast_plate_ocr.cli.visualize_predictions import visualize_predictions
except ImportError as e:
    raise ImportError("Make sure to 'pip install fast-plate-ocr[train]' to run this!") from e


@click.group(context_settings={"max_content_width": 120})
def main_cli():
    """FastPlateOCR CLI."""


main_cli.add_command(visualize_predictions)
main_cli.add_command(visualize_augmentation)
main_cli.add_command(valid)
main_cli.add_command(train)
main_cli.add_command(export_onnx)
