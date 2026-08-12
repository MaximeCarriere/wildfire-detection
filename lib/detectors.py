"""Runtime adapters — the things :mod:`lib.evaluator` measures.

Every runtime in the series (PyTorch FP16, ONNX, TensorRT FP16 / INT8 / sparse
INT8) shows up here behind the same tiny interface, so the harness never learns
which one it is holding and the numbers stay comparable across Phase 1-4.

Two deliberate asymmetries, both of which would otherwise quietly distort the
series:

* **Accuracy runs threshold-free** (conf 0.001) because COCO-style mAP sweeps the
  whole confidence range. **Latency runs at the deployment threshold**
  (conf 0.25) because NMS cost scales with how many candidate boxes survive
  filtering — timing a 0.001 threshold would measure a configuration nobody
  would ever deploy, and would penalise models that produce more low-confidence
  noise for a reason unrelated to their speed.
* **Timed scope is forward + NMS**, with image decode and letterboxing hoisted
  out of the loop. A fire watch needs boxes, not logits, so NMS belongs in the
  number; JPEG decode is a property of the camera pipeline, not the model.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Sequence

from lib.evaluator import DEMO_CONF, EVAL_CONF, MAX_DET, NMS_IOU, ImageDets

#: Where the original YOLOv5 repository is checked out. Needed only by
#: :class:`Yolov5Detector` — see its docstring for why.
YOLOV5_REPO = Path(os.environ.get("YOLOV5_REPO", Path.home() / "yolov5"))


class UltralyticsDetector:
    """YOLOv8 in PyTorch — the baseline runtime for Phases 0-2 (PLAN.md XP1).

    FP16 by default: PLAN.md fixes "PyTorch FP16" as the baseline runtime, stated
    once and used everywhere until TensorRT lands in Phase 3. Ultralytics 8.4
    replaced the ``half`` flag with a unified ``quantize`` scheme (16 = FP16),
    which is what the predict path passes.
    """

    def __init__(self, weights: str | Path, *, input_res: int = 640, device: str = "cuda:0",
                 half: bool = True, name: str | None = None):
        from ultralytics import YOLO

        self.weights = Path(weights)
        self.input_res = input_res
        self.device = device
        self.half = half
        self.name = name or self.weights.stem
        self.model = YOLO(str(weights))
        self.model.to(device)
        self.torch_model = self.model.model
        if half:
            self.torch_model = self.torch_model.half()
        self.torch_model.eval()
        self._nc = len(self.model.names)

    # ---- accuracy path ---------------------------------------------------

    def predict(self, images: Sequence[Path], *, batch: int = 16) -> list[ImageDets]:
        """Threshold-free detections in original-image pixel coordinates.

        Delegates to Ultralytics' own predict path: it owns the letterbox and the
        rescale back to source resolution, and re-implementing that is exactly
        the kind of per-XP divergence the frozen protocol exists to prevent.
        """
        out: list[ImageDets] = []
        images = [Path(p) for p in images]
        for i in range(0, len(images), batch):
            chunk = [str(p) for p in images[i:i + batch]]
            results = self.model.predict(
                chunk, imgsz=self.input_res, conf=EVAL_CONF, iou=NMS_IOU,
                max_det=MAX_DET, quantize=16 if self.half else None,
                device=self.device, verbose=False,
            )
            for path, r in zip(images[i:i + batch], results):
                b = r.boxes
                out.append(ImageDets(
                    stem=path.stem,
                    xyxy=[tuple(map(float, xy)) for xy in b.xyxy.tolist()],
                    scores=[float(c) for c in b.conf.tolist()],
                    classes=[int(c) for c in b.cls.tolist()],
                ))
        return out

    # ---- timing path -----------------------------------------------------

    def prepare_frames(self, images: Sequence[Path]) -> list:
        """Decode + letterbox once, up front, and park the tensors on the device.

        Everything here is deliberately outside the timed loop.
        """
        import cv2
        import torch
        from ultralytics.data.augment import LetterBox

        lb = LetterBox((self.input_res, self.input_res), auto=False, stride=32)
        frames = []
        for p in images:
            im = cv2.imread(str(p))
            if im is None:
                continue
            im = lb(image=im)                                  # HWC BGR, padded to square
            t = torch.from_numpy(im[..., ::-1].transpose(2, 0, 1).copy())   # -> CHW RGB
            t = t.to(self.device).float().div_(255.0).unsqueeze(0)
            frames.append(t.half() if self.half else t)
        return frames

    def infer_frame(self, frame) -> None:
        """One batch-1 detection: forward + NMS, device-synchronized.

        The synchronize is what makes the per-frame number honest — without it,
        CUDA's asynchrony would let work spill past the timer and show up as
        someone else's latency.
        """
        import torch
        # Ultralytics 8.4 moved NMS out of utils.ops into its own module; import
        # here (not at module scope) so the pin is visible at the call site.
        from ultralytics.utils.nms import non_max_suppression

        with torch.inference_mode():
            pred = self.torch_model(frame)
            pred = pred[0] if isinstance(pred, (list, tuple)) else pred
            non_max_suppression(
                pred.float(), conf_thres=DEMO_CONF, iou_thres=NMS_IOU,
                max_det=MAX_DET, nc=self._nc,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def infer_batch(self, batch) -> None:
        """Forward pass on a stacked batch — the compute-bound measurement.

        No NMS: at batch 16 with a threshold-free head, NMS would dominate for
        reasons that have nothing to do with the model's compute cost, which is
        what this number exists to expose."""
        import torch

        with torch.inference_mode():
            self.torch_model(batch)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    # ---- model facts for the results schema ------------------------------

    def params_m(self) -> float:
        return sum(p.numel() for p in self.torch_model.parameters()) / 1e6

    def size_disk_mb(self) -> float:
        return self.weights.stat().st_size / 1e6


class Yolov5Detector:
    """Original-YOLOv5-repo checkpoints — specifically the D-Fire authors' own
    published `yolov5s.pt` / `yolov5l.pt` (github.com/pedbrgs/Fire-Detection, MIT).

    These need their own adapter for a reason worth recording, because it is a
    trap that fails *silently*: the ``ultralytics`` package will not load an
    original YOLOv5 checkpoint, and rather than erroring it **substitutes its
    own**. Passing ``weights/yolov5s.pt`` to ``YOLO()`` makes it recognise the
    filename as a standard model name, download COCO-pretrained ``yolov5su.pt``
    from the Ultralytics asset server, and return that — 80 COCO classes, none of
    them fire. Any accuracy measured that way would be a real number about
    entirely the wrong model.

    So the checkpoint is unpickled against the actual YOLOv5 codebase (cloned to
    :data:`YOLOV5_REPO`), which is what its saved classes resolve against. The
    class order in these weights is ``['smoke', 'fire']`` — index 0 smoke, 1 fire,
    matching the mapping XP0 proved against D-Fire's published box counts, so no
    remapping is needed. The constructor asserts it anyway.

    Note these are anchor-based YOLOv5, not the anchor-free "u" variants; the
    detection head and NMS come from the YOLOv5 repo, while the *protocol*
    (thresholds, max_det, timed scope) stays identical to
    :class:`UltralyticsDetector` so the two are comparable.
    """

    EXPECTED_NAMES = ["smoke", "fire"]

    def __init__(self, weights: str | Path, *, input_res: int = 640, device: str = "cuda:0",
                 half: bool = True, name: str | None = None, repo: Path = YOLOV5_REPO):
        import torch

        repo = Path(repo)
        if not (repo / "models" / "yolo.py").exists():
            raise FileNotFoundError(
                f"YOLOv5 repo not found at {repo}. These checkpoints can only be unpickled "
                f"against it: git clone --depth 1 https://github.com/ultralytics/yolov5 {repo} "
                f"(or set YOLOV5_REPO).")
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))

        self.weights = Path(weights)
        self.input_res = input_res
        self.device = device
        self.half = half
        self.name = name or self.weights.stem

        # weights_only=False is required: the checkpoint stores the live model
        # object, not a state_dict. These files come from the dataset authors and
        # were fetched by hand; treat any other source as untrusted.
        ckpt = torch.load(self.weights, map_location="cpu", weights_only=False)
        model = (ckpt.get("ema") or ckpt["model"]).float()
        model.eval().to(device)
        if half:
            model.half()
        self.model = model

        names = list(model.names.values()) if isinstance(model.names, dict) else list(model.names)
        if names != self.EXPECTED_NAMES:
            raise ValueError(
                f"{self.weights} has class names {names}, expected {self.EXPECTED_NAMES}. "
                f"Every per-class number in the series assumes 0=smoke, 1=fire — refusing "
                f"rather than silently mislabelling the classes.")
        self.names = names

    # ---- accuracy path ---------------------------------------------------

    def predict(self, images: Sequence[Path], *, batch: int = 16) -> list[ImageDets]:
        """Threshold-free detections in original-image pixel coordinates."""
        import cv2
        import torch
        from utils.augmentations import letterbox
        from utils.general import non_max_suppression, scale_boxes

        out: list[ImageDets] = []
        images = [Path(p) for p in images]
        for i in range(0, len(images), batch):
            chunk = images[i:i + batch]
            tensors, shapes = [], []
            for p in chunk:
                im0 = cv2.imread(str(p))
                if im0 is None:
                    raise FileNotFoundError(f"could not read {p}")
                im = letterbox(im0, (self.input_res, self.input_res), stride=32, auto=False)[0]
                tensors.append(torch.from_numpy(im[..., ::-1].transpose(2, 0, 1).copy()))
                shapes.append(im0.shape[:2])

            x = torch.stack(tensors).to(self.device).float().div_(255.0)
            if self.half:
                x = x.half()

            with torch.inference_mode():
                pred = self.model(x)
                pred = pred[0] if isinstance(pred, (list, tuple)) else pred
                dets = non_max_suppression(pred.float(), conf_thres=EVAL_CONF,
                                           iou_thres=NMS_IOU, max_det=MAX_DET)

            # scale_boxes mutates in place, which torch forbids on tensors born
            # inside inference mode. Cloning *outside* the block is what promotes
            # them back to normal tensors — a clone taken inside stays an
            # inference tensor and fails identically.
            dets = [d.clone() for d in dets]

            for path, det, shape0 in zip(chunk, dets, shapes):
                if len(det):
                    det[:, :4] = scale_boxes(x.shape[2:], det[:, :4], shape0).round()
                det = det.cpu()
                out.append(ImageDets(
                    stem=path.stem,
                    xyxy=[tuple(map(float, row[:4])) for row in det],
                    scores=[float(row[4]) for row in det],
                    classes=[int(row[5]) for row in det],
                ))
        return out

    # ---- timing path -----------------------------------------------------

    def prepare_frames(self, images: Sequence[Path]) -> list:
        """Decode + letterbox once, up front — deliberately outside the timed loop."""
        import cv2
        import torch
        from utils.augmentations import letterbox

        frames = []
        for p in images:
            im0 = cv2.imread(str(p))
            if im0 is None:
                continue
            im = letterbox(im0, (self.input_res, self.input_res), stride=32, auto=False)[0]
            t = torch.from_numpy(im[..., ::-1].transpose(2, 0, 1).copy())
            t = t.to(self.device).float().div_(255.0).unsqueeze(0)
            frames.append(t.half() if self.half else t)
        return frames

    def infer_frame(self, frame) -> None:
        """One batch-1 detection: forward + NMS, device-synchronized. Same scope
        and same thresholds as :meth:`UltralyticsDetector.infer_frame`."""
        import torch
        from utils.general import non_max_suppression

        with torch.inference_mode():
            pred = self.model(frame)
            pred = pred[0] if isinstance(pred, (list, tuple)) else pred
            non_max_suppression(pred.float(), conf_thres=DEMO_CONF, iou_thres=NMS_IOU,
                                max_det=MAX_DET)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def infer_batch(self, batch) -> None:
        """Forward pass on a stacked batch — the compute-bound measurement.
        No NMS, for the same reason as :meth:`UltralyticsDetector.infer_batch`."""
        import torch

        with torch.inference_mode():
            self.model(batch)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    # ---- model facts for the results schema ------------------------------

    def params_m(self) -> float:
        return sum(p.numel() for p in self.model.parameters()) / 1e6

    def size_disk_mb(self) -> float:
        return self.weights.stat().st_size / 1e6


class TRTDetector:
    """A TensorRT engine (FP16 / INT8 / sparse INT8) behind the same interface.

    Everything outside the engine itself is shared with :class:`Yolov5Detector` —
    the same letterbox, the same NMS, the same thresholds, the same timed scope.
    That is deliberate: if the pre/post-processing differed, a TRT-vs-PyTorch
    comparison would be measuring the wrapper as much as the runtime, and the
    whole Phase-3 speedup claim would be contaminated.

    The engines are built with a dynamic batch profile, so one engine answers
    both questions XP2 showed are different: batch-1 latency (what one camera
    feed sees) and batched throughput (what the hardware can do).
    """

    def __init__(self, engine: str | Path, *, input_res: int, fmt: str,
                 params_m: float, source_weights: str | Path | None = None,
                 device: str = "cuda:0", name: str | None = None,
                 repo: Path = YOLOV5_REPO):
        from lib.trt_runner import TRTModel

        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))

        self.engine_path = Path(engine)
        self.input_res = input_res
        self.device = device
        self.fmt = fmt
        self.name = name or self.engine_path.stem
        self.model = TRTModel(self.engine_path)
        self._in_dtype = self.model.input_dtype()
        # Parameter count is a property of the network, not the engine, so it is
        # carried over from the source checkpoint rather than invented.
        self._params_m = params_m
        self._source_weights = Path(source_weights) if source_weights else None

    # ---- accuracy path ---------------------------------------------------

    def predict(self, images: Sequence[Path], *, batch: int = 8) -> list[ImageDets]:
        import cv2
        import torch
        from utils.augmentations import letterbox
        from utils.general import non_max_suppression, scale_boxes

        out: list[ImageDets] = []
        images = [Path(p) for p in images]
        for i in range(0, len(images), batch):
            chunk = images[i:i + batch]
            tensors, shapes = [], []
            for p in chunk:
                im0 = cv2.imread(str(p))
                if im0 is None:
                    raise FileNotFoundError(f"could not read {p}")
                im = letterbox(im0, (self.input_res, self.input_res), stride=32, auto=False)[0]
                tensors.append(torch.from_numpy(im[..., ::-1].transpose(2, 0, 1).copy()))
                shapes.append(im0.shape[:2])

            x = torch.stack(tensors).to(self.device).to(self._in_dtype).div_(255.0)
            pred = self.model.infer(x)
            dets = non_max_suppression(pred.float(), conf_thres=EVAL_CONF,
                                       iou_thres=NMS_IOU, max_det=MAX_DET)

            for path, det, shape0 in zip(chunk, dets, shapes):
                det = det.clone()
                if len(det):
                    det[:, :4] = scale_boxes(x.shape[2:], det[:, :4], shape0).round()
                det = det.cpu()
                out.append(ImageDets(
                    stem=path.stem,
                    xyxy=[tuple(map(float, row[:4])) for row in det],
                    scores=[float(row[4]) for row in det],
                    classes=[int(row[5]) for row in det],
                ))
        return out

    # ---- timing path -----------------------------------------------------

    def prepare_frames(self, images: Sequence[Path]) -> list:
        import cv2
        import torch
        from utils.augmentations import letterbox

        frames = []
        for p in images:
            im0 = cv2.imread(str(p))
            if im0 is None:
                continue
            im = letterbox(im0, (self.input_res, self.input_res), stride=32, auto=False)[0]
            t = torch.from_numpy(im[..., ::-1].transpose(2, 0, 1).copy())
            t = t.to(self.device).to(self._in_dtype).div_(255.0).unsqueeze(0)
            frames.append(t)
        return frames

    def infer_frame(self, frame) -> None:
        """Batch-1 detection: engine + NMS. TRTModel.infer already synchronizes
        the stream, so the timer sees completed work."""
        from utils.general import non_max_suppression

        pred = self.model.infer(frame)
        non_max_suppression(pred.float(), conf_thres=DEMO_CONF, iou_thres=NMS_IOU,
                            max_det=MAX_DET)

    def infer_batch(self, batch) -> None:
        """Engine forward on a stacked batch, no NMS — the compute-bound number."""
        self.model.infer(batch)

    # ---- model facts for the results schema ------------------------------

    def params_m(self) -> float:
        return self._params_m

    def size_disk_mb(self) -> float:
        """The *engine* on disk — the number that matters for deployment, and the
        one that changes under quantization."""
        return self.engine_path.stat().st_size / 1e6


__all__ = ["UltralyticsDetector", "Yolov5Detector", "TRTDetector", "YOLOV5_REPO"]
