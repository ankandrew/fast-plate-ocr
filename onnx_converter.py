"""
Script to convert Keras model to ONNX.
"""
from argparse import ArgumentParser

import keras2onnx
import tensorflow as tf

from fast_lp_ocr.custom import cat_acc, cce, plate_acc, top_3_k


def args_parser():
    parser = ArgumentParser(description="Convert keras -> onnx.")
    parser.add_argument("--model", default="models/m1_93_vpa_2.0M-i2.h5", help="Path to h5 model")
    parser.add_argument("--output", help="Output filepath for onnx model")
    return parser.parse_args()


if __name__ == "__main__":
    args = args_parser()
    custom_objects = {"cce": cce, "cat_acc": cat_acc, "plate_acc": plate_acc, "top_3_k": top_3_k}
    model = tf.keras.models.load_model(args.model, custom_objects=custom_objects)
    onnx_model = keras2onnx.convert_keras(model, model.name)
    out_file_path = (
        args.output if args.output else args.model.split("/")[-1].rstrip(".h5") + ".onnx"
    )
    keras2onnx.save_model(onnx_model, out_file_path)
