# gst-rknninference

A GStreamer plugin that runs `.rknn` models on Rockchip NPUs through
RKNN Runtime. `rknninference` passes video frames through unchanged while
attaching raw RKNN output tensors as metadata; optional decoder elements turn
those tensors into GStreamer Analytics classification or detection metadata.

The plugin supports RK3588 and RK3576 targets plus system-memory and DMA-BUF
video frames. Its regular raw-video caps advertise RGB, BGR, NV12, and NV21.
With RGA, NV12 and NV21 inputs are converted to RGB; the CPU fallback accepts
only RGB and BGR. DMA_DRM caps are supported with GStreamer 1.24 or newer.

## Requirements

- Meson >= 0.62 and a C11 compiler
- GStreamer development packages >= 1.24:
  `gstreamer-1.0`, `gstreamer-base-1.0`, `gstreamer-video-1.0`, and
  `gstreamer-allocators-1.0`
- GLib/GIO development files
- Rockchip RKNN Runtime (`librknnrt` and headers, expected under
  `/usr/local/include`)

Optional dependencies:

- `gstreamer-analytics-1.0` enables the tensor decoder elements
- `librga` (or `librga` discoverable as `rga`) enables RGA hardware
  preprocessing

## Build and install

```sh
meson setup build
meson compile -C build
sudo meson install -C build
```

For a non-system installation, point GStreamer at the installed plugin:

```sh
export GST_PLUGIN_PATH=/path/to/install/lib/gstreamer-1.0${GST_PLUGIN_PATH:+:$GST_PLUGIN_PATH}
gst-inspect-1.0 rknninference
```

The plugin is installed as `libgstrknninference.so`. Confirm which optional
elements were built with:

```sh
gst-inspect-1.0 rknninference
```

## Elements

| Element | Availability | Purpose |
|---|---|---|
| `rknninference` | Always | Runs a `.rknn` model and attaches raw output tensor metadata to each frame. |
| `rknnyolov8tensordec` | GStreamer Analytics | Decodes YOLOv8, YOLOv10, and YOLOv11 DFL outputs into object-detection metadata. |
| `rknnyolov5tensordec` | GStreamer Analytics | Decodes anchor-based YOLOv5 outputs into object-detection metadata. |
| `rknnpicodettensordec` | GStreamer Analytics | Decodes raw PP-PicoDet outputs (`export.post_process=False`) into object-detection metadata. |
| `rknnclassifiertensordec` | GStreamer Analytics | Produces top-K classification metadata. |

`rknninference` properties:

- `model`: required path to the `.rknn` model
- `npu-core`: NPU core mask; `0` automatically selects a core, while `1`,
  `2`, `4`, and `7` select core 0, 1, 2, or all three
- `inference-interval`: run inference every N frames (default `1`)

The object-detection decoders expose `class-confidence-threshold`,
`iou-threshold`, `max-detections`, and `labels-file`. PicoDet also has
`num-classes` and `reg-max`. The classifier decoder exposes
`class-confidence-threshold`, `top-k`, and `labels-file`.

`rknnpicodettensordec` expects raw PicoDet tensor outputs. Export PicoDet
models from PaddlePaddle with `export.post_process=False`, or export with
LibreYOLO after applying the bundled patch and setting `paddle_output=True`.

## Pipelines

Run inference on a test source:

```sh
gst-launch-1.0 videotestsrc ! video/x-raw,format=RGB,width=640,height=480 \
  ! rknninference model=/path/to/model.rknn ! fakesink
```

Decode YOLOv8-family results:

```sh
gst-launch-1.0 filesrc location=input.mp4 ! decodebin ! videoconvert \
  ! video/x-raw,format=RGB ! rknninference model=/path/to/yolo.rknn \
  ! rknnyolov8tensordec labels-file=/path/to/labels.txt ! fakesink
```

Replace `rknnyolov8tensordec` with the decoder matching the model output.
The decoder adds analytics metadata; a downstream element must consume or
render that metadata if visual annotations are required.

## Model conversion

`scripts/convert_onnx_to_rknn.py` is a helper for converting compatible
static-shape ONNX models with RKNN Toolkit2. It supports a target platform,
calibration dataset, INT8 quantization controls, configurable input
normalization, and explicit input dimensions. Install RKNN Toolkit2 in the
Python environment that runs the script:

```sh
python3 scripts/convert_onnx_to_rknn.py --help
```

The repository also includes
`scripts/libreyolo-picodet-rectangular-paddle-output.patch`, which adds
LibreYOLO export options needed for raw PicoDet output and rectangular input
models. Rectangular inputs such as 416x256 are supported by the inference
element when encoded in the model.

## License

LGPL-2.1-or-later.
