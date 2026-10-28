#!/usr/bin/env python3
"""Convert a static-shape ONNX model to RKNN for RK35xx series."""

import argparse
import os
import sys

try:
    from rknn.api import RKNN
except ModuleNotFoundError as error:
    if error.name == "rknn":
        raise SystemExit(
            "RKNN Toolkit2 is not installed. Install its rknn_toolkit2 "
            "wheel, then rerun this script."
        ) from error
    raise
def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("onnx", help="Input ONNX model")
    parser.add_argument("output", help="Output RKNN model")
    parser.add_argument(
        "--dataset",
        help="Calibration dataset text file, one image path per line",
    )
    parser.add_argument(
        "--target",
        default="rk3576",
        help="RKNN target platform (default: rk3576)",
    )
    parser.add_argument(
        "--no-quant",
        action="store_true",
        help="Build without INT8 quantization",
    )
    parser.add_argument(
        "--quantized-algorithm",
        choices=("normal", "mmse", "kl_divergence", "gdq"),
        default="normal",
        help="INT8 calibration algorithm (default: normal)",
    )
    parser.add_argument(
        "--quantized-method",
        default="channel",
        help="INT8 quantization method: layer, channel, or group32..group256",
    )
    parser.add_argument(
        "--quantized-hybrid-level",
        type=int,
        choices=(0, 1, 2),
        default=0,
        help="Enable RKNN hybrid quantization level (default: 0)",
    )
    parser.add_argument(
        "--mean",
        nargs=3,
        type=float,
        default=[123.675, 116.28, 103.53],
        metavar=("R", "G", "B"),
        help="Input mean values (default: 123.675 116.28 103.53)",
    )
    parser.add_argument(
        "--std",
        nargs=3,
        type=float,
        default=[58.395, 57.12, 57.375],
        metavar=("R", "G", "B"),
        help="Input standardization values (default: 58.395 57.12 57.375)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=320,
        help="Input width of the model (default: 320)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=320,
        help="Input height of the model (default: 320)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.isfile(args.onnx):
        print(f"ONNX model not found: {args.onnx}", file=sys.stderr)
        return 2

    quantize = not args.no_quant
    if quantize and not args.dataset:
        print("--dataset is required unless --no-quant is used", file=sys.stderr)
        return 2
    if quantize and not os.path.isfile(args.dataset):
        print(f"Calibration dataset not found: {args.dataset}", file=sys.stderr)
        return 2

    rknn = RKNN(verbose=True)
    try:

        print("Configuring RKNN...")
        ret = rknn.config(
            mean_values=[args.mean],
            std_values=[args.std],
            target_platform=args.target,
            quantized_dtype="w8a8",
            quantized_algorithm=args.quantized_algorithm,
            quantized_method=args.quantized_method,
            quantized_hybrid_level=args.quantized_hybrid_level,
            optimization_level=3,
            quant_img_RGB2BGR=False,
        )
        if ret != 0:
            raise RuntimeError(f"rknn.config failed: {ret}")

        print(f"Loading {args.onnx} with shape [1, 3, {args.height}, {args.width}]...")
        ret = rknn.load_onnx(
            model=args.onnx,
            input_size_list=[[1, 3, args.height, args.width]]
        )

        if ret != 0:
            raise RuntimeError(f"rknn.load_onnx failed: {ret}")

        print(f"Building ({'INT8' if quantize else 'unquantized'})...")
        ret = rknn.build(
            do_quantization=quantize,
            dataset=args.dataset if quantize else None,
        )
        if ret != 0:
            raise RuntimeError(f"rknn.build failed: {ret}")

        print(f"Exporting {args.output}...")
        ret = rknn.export_rknn(args.output)
        if ret != 0:
            raise RuntimeError(f"rknn.export_rknn failed: {ret}")
    finally:
        rknn.release()

    print("Conversion complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
