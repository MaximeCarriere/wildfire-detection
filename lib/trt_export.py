"""ONNX -> TensorRT engine building, including INT8 post-training quantization.

FP16 engines are built with ``trtexec`` (see scripts/build_fp16.sh) because there
is nothing to configure. INT8 needs a calibrator fed with *our* frozen 500-image
calibration set, which is a Python-API job — and getting that set right is the
whole point of XP0 having frozen it: calibration data determines the activation
ranges, so a calibration set that omits night and backlight frames produces an
engine that quietly falls apart on exactly those conditions. XP10's slice
analysis is what checks whether it worked.

The calibrator preprocesses images through the **same letterbox path as
inference** (:class:`lib.detectors.Yolov5Detector`). A mismatch there would
calibrate the ranges against a distribution the model never actually sees.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _letterbox_batch(paths, res: int, repo: Path):
    """Preprocess a list of images exactly as inference does -> (B,3,res,res) float32."""
    import cv2
    import numpy as np

    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from utils.augmentations import letterbox

    ims = []
    for p in paths:
        im0 = cv2.imread(str(p))
        if im0 is None:
            raise FileNotFoundError(p)
        im = letterbox(im0, (res, res), stride=32, auto=False)[0]
        ims.append(im[..., ::-1].transpose(2, 0, 1).copy())        # BGR->RGB, CHW
    return np.ascontiguousarray(np.stack(ims), dtype=np.float32) / 255.0


def make_calibrator(image_paths, res: int, yolov5_repo: Path, cache_path: Path,
                    batch_size: int = 8, algorithm: str = "entropy"):
    """A calibrator over the frozen calibration set.

    ``algorithm`` selects between TensorRT's two, and the choice turns out to
    matter enormously here:

    * ``"entropy"`` — IInt8EntropyCalibrator2, TensorRT's default. Picks ranges by
      minimising KL divergence between the quantized and full-precision
      distributions, deliberately **clipping outliers**. Tuned for classification.
    * ``"minmax"`` — IInt8MinMaxCalibrator. Uses the observed min/max, clipping
      nothing. NVIDIA recommends it for detection and segmentation.

    **Measured, XP10.** The entropy cache is decodable (``scale = max_abs / 127``;
    the anchor-grid constants confirm it — 0.00393797 x 127 = 0.5 exactly), and it
    shows entropy calibration assigning the *input tensor* a range of
    ``0.0035237 x 127 = 0.4475``. The input is normalised to [0,1] and every
    daylight frame contains sky near 1.0 — verified, the calibrator is fed data
    with max exactly 1.0 and preprocessing is bit-identical to inference. So
    entropy clips the top 55% of the input range to saturation, destroying exactly
    the bright-sky contrast that faint smoke lives in.
    """
    import tensorrt as trt
    import torch

    base = {"entropy": trt.IInt8EntropyCalibrator2,
            "minmax": trt.IInt8MinMaxCalibrator}[algorithm]

    class Calibrator(base):
        def __init__(self):
            super().__init__()
            self.paths = [Path(p) for p in image_paths]
            self.res = res
            self.batch_size = batch_size
            self.idx = 0
            self.cache = Path(cache_path)
            # One reusable device buffer; TRT only needs the pointer to be valid
            # for the duration of the get_batch call.
            self.device_input = torch.empty(
                (batch_size, 3, res, res), dtype=torch.float32, device="cuda")

        def get_batch_size(self):
            return self.batch_size

        def get_batch(self, names):
            if self.idx + self.batch_size > len(self.paths):
                return None                      # signals "calibration data exhausted"
            chunk = self.paths[self.idx:self.idx + self.batch_size]
            self.idx += self.batch_size
            arr = _letterbox_batch(chunk, self.res, yolov5_repo)
            self.device_input.copy_(torch.from_numpy(arr))
            if self.idx % (self.batch_size * 10) == 0:
                print(f"  calibrated {self.idx}/{len(self.paths)}", flush=True)
            return [int(self.device_input.data_ptr())]

        def read_calibration_cache(self):
            return self.cache.read_bytes() if self.cache.exists() else None

        def write_calibration_cache(self, cache):
            self.cache.write_bytes(cache)

    return Calibrator()


def build_int8_engine(onnx_path: Path, engine_path: Path, *, calibrator,
                      res: int, max_batch: int = 16,
                      workspace_gb: float = 3.0, fp16_head: bool = True,
                      fp16_head_convs: int = 3) -> Path:
    """Build an INT8 engine with a dynamic batch profile.

    FP16 is left enabled alongside INT8: TensorRT falls back to FP16 for layers
    where INT8 would be catastrophic or unsupported, which is the standard and
    honest configuration — a pure-INT8 engine is not what anyone deploys.

    ``fp16_head`` (default on) pins everything after the last convolution to
    FP16, and it is not optional in practice. **Measured, XP10:** without it,
    YOLOv5s INT8 scored mAP50 0.2554 against FP16's 0.7776 — a 67% collapse, with
    tiny-plume accuracy down 99% (0.1376 -> 0.0016) while classification partly
    survived (the night slice held 0.48).

    The cause is visible in the exported graph. YOLOv5's Detect layer decodes raw
    head outputs into a single tensor that concatenates **box coordinates in
    pixels (0-640)** with **objectness and class probabilities in [0,1]**. INT8
    carries one scale per tensor, so 256 levels are stretched across 0-640 —
    about 2.5 px of granularity for every box edge — while the probabilities are
    squeezed into a fraction of a single quantization step. Box regression is
    destroyed; the classifier limps on. That asymmetry is the fingerprint.

    Pinning the decode tail alone was **not sufficient**: it lifted mAP50 from
    0.5211 to only about half of FP16's 0.9285 on a fixed sanity subset, with
    tiny-plume still ~99% gone. ``fp16_head_convs`` therefore also pins the last
    N convolutions — YOLOv5's three per-stride Detect convs, which emit the raw
    box/objectness/class channels. Those channels have wildly different scales
    within a single output tensor, so an INT8 per-tensor scale damages box
    regression at the source, before the decode ever runs. Together the two
    constraints are the standard "quantize the backbone and neck, keep the
    detection head in FP16" recipe.

    All of it is a tiny share of total FLOPs, so the accuracy is bought back for
    almost no speed — which XP10 measures rather than assumes.
    """
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            errs = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
            raise RuntimeError(f"ONNX parse failed:\n{errs}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gb * (1 << 30)))
    config.set_flag(trt.BuilderFlag.INT8)
    config.set_flag(trt.BuilderFlag.FP16)
    config.int8_calibrator = calibrator

    if fp16_head:
        # Everything after the last convolution is the Detect decode (sigmoid,
        # anchor/stride arithmetic, concat); the last few convolutions are the
        # Detect head itself. Both are pinned to FP16 — see the docstring for the
        # measured cost of pinning neither, and of pinning only the tail.
        conv_idx = [i for i in range(network.num_layers)
                    if network.get_layer(i).type == trt.LayerType.CONVOLUTION]
        if not conv_idx:
            raise RuntimeError("no convolution found — is this the right ONNX?")
        last_conv = conv_idx[-1]

        head_convs = conv_idx[-fp16_head_convs:] if fp16_head_convs > 0 else []
        for i in head_convs:
            layer = network.get_layer(i)
            layer.precision = trt.float16
            for o in range(layer.num_outputs):
                layer.set_output_type(o, trt.float16)
        if head_convs:
            print(f"  pinned {len(head_convs)} Detect head convolutions to FP16 "
                  f"(layer indices {head_convs})", flush=True)
        config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
        float_types = (trt.DataType.FLOAT, trt.DataType.HALF)
        pinned = skipped = 0
        for i in range(last_conv + 1, network.num_layers):
            layer = network.get_layer(i)
            outs = [layer.get_output(o) for o in range(layer.num_outputs)]
            # Judge a layer by its OUTPUTS only. An earlier version also rejected
            # any layer with a shape-tensor *input*, which was wrong and quietly
            # defeated the whole fix: a Reshape takes (data, shape), so every
            # reshape carrying box coordinates was skipped and left in INT8. It
            # skipped 166 layers and recovered mAP50 only from 0.5211 to 0.5387
            # against FP16's 0.9285. What must be avoided is pinning layers that
            # *produce* shapes/indices — TensorRT rejects those outright ("cannot
            # use precision Half for layer that computes indices") — and that is
            # exactly what the output test catches.
            if not outs or any(t.is_shape_tensor for t in outs) \
                    or any(t.dtype not in float_types for t in outs):
                skipped += 1
                continue
            try:
                layer.precision = trt.float16
                for o in range(layer.num_outputs):
                    layer.set_output_type(o, trt.float16)
                pinned += 1
            except Exception:
                skipped += 1          # some layers refuse a constraint; leave them
        print(f"  pinned {pinned} decode layers to FP16, skipped {skipped} shape/index "
              f"layers (range {last_conv + 1}..{network.num_layers - 1})", flush=True)

    in_name = network.get_input(0).name
    profile = builder.create_optimization_profile()
    profile.set_shape(in_name, (1, 3, res, res), (1, 3, res, res), (max_batch, 3, res, res))
    config.add_optimization_profile(profile)

    # Calibration runs at a single shape; without this TRT complains for dynamic
    # networks. The calibrator's batch size must match the profile it is given.
    calib_profile = builder.create_optimization_profile()
    bs = calibrator.get_batch_size()
    calib_profile.set_shape(in_name, (bs, 3, res, res), (bs, 3, res, res), (bs, 3, res, res))
    config.set_calibration_profile(calib_profile)

    print(f"building INT8 engine {engine_path.name} (this takes a while) …", flush=True)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("INT8 engine build failed")
    engine_path.write_bytes(serialized)
    print(f"wrote {engine_path} ({engine_path.stat().st_size / 1e6:.1f} MB)", flush=True)
    return engine_path


__all__ = ["make_calibrator", "build_int8_engine"]
