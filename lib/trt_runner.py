"""Minimal TensorRT runtime for the exported YOLOv5 detection engines.

Runs an engine directly on torch CUDA tensors (zero-copy via ``data_ptr``, no
pycuda). Ported in spirit from jetson-xray-panel/lib/trt_runner.py, extended for
dynamic shapes: the detection engines are built with a batch profile so one
engine serves both the batch-1 latency measurement and the batched
compute-bound throughput number that XP2 showed is the only meaningful speed
axis in PyTorch.

TensorRT 10.3 API (JetPack R36.4.3): named IO tensors, ``set_input_shape`` +
``set_tensor_address`` + ``execute_async_v3``.
"""
from __future__ import annotations

from pathlib import Path


class TRTModel:
    """One deserialized engine, callable on torch CUDA tensors."""

    def __init__(self, engine_path: str | Path):
        import tensorrt as trt

        self.path = Path(engine_path)
        self.logger = trt.Logger(trt.Logger.ERROR)
        with open(self.path, "rb") as f, trt.Runtime(self.logger) as rt:
            self.engine = rt.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"failed to deserialize {self.path}")
        self.context = self.engine.create_execution_context()

        self.in_name = self.out_name = None
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.in_name = name
            else:
                # YOLOv5 exports several outputs; the first is the concatenated
                # inference tensor (B, N, 5+nc) that NMS consumes. The rest are
                # per-stride training outputs we do not need.
                if self.out_name is None:
                    self.out_name = name

    def infer(self, x):
        """x: (B,3,H,W) float32/float16 CUDA contiguous -> (B, N, 5+nc) float32 CUDA."""
        import torch

        if not x.is_contiguous():
            x = x.contiguous()
        self.context.set_input_shape(self.in_name, tuple(x.shape))
        out_shape = tuple(self.context.get_tensor_shape(self.out_name))
        out = torch.empty(out_shape, dtype=torch.float32, device="cuda")

        self.context.set_tensor_address(self.in_name, x.data_ptr())
        self.context.set_tensor_address(self.out_name, out.data_ptr())
        stream = torch.cuda.current_stream()
        self.context.execute_async_v3(stream.cuda_stream)
        stream.synchronize()
        return out

    def input_dtype(self):
        """Engines built with --fp16 still take FP32 input unless the network's
        input tensor itself was set to half; ask rather than assume."""
        import tensorrt as trt
        import torch

        dt = self.engine.get_tensor_dtype(self.in_name)
        return torch.float16 if dt == trt.DataType.HALF else torch.float32


__all__ = ["TRTModel"]
