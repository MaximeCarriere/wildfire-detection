"""PlatformIO pre-build hook: give ESP-NN's scratch buffers 16-byte alignment.

The ESP32-S3 kernels load 128 bits at a time and require their scratch buffer to
be 16-byte aligned. ``heap_caps_malloc`` only promises 8, and on this board it
returned 0x3fcec818 -- 8 bytes off. The kernel rounded down to 0x3fcec810, which
is exactly where the allocator keeps that block's header, and the next attempt to
grow the buffer aborted with::

    CORRUPT HEAP: Bad head at 0x3fcec810. Expected 0xabba1234 got 0x00090009

**Latent until the kernels became real.** The ANSI reference implementations are
plain C and do not care about alignment, so every earlier run passed. Turning on
CONFIG_NN_OPTIMIZED is what made a misaligned buffer start corrupting the heap,
which is a good argument for treating a large speed-up as a reason to re-check
correctness rather than as a result on its own.

Rewrites both allocation sites -- the fork's conv.cpp and the depthwise kernel
patched in by patch_depthwise.py -- to use ``heap_caps_aligned_alloc``. The size is
also rounded up to a multiple of the alignment, which aligned_alloc is entitled to
require and which costs at most fifteen bytes.

Idempotent, and it leaves the file alone if the allocation does not look as
expected rather than guessing.
"""

from pathlib import Path

Import("env")  # noqa: F821  -- injected by PlatformIO

OLD = "heap_caps_malloc(needed, MALLOC_CAP_8BIT)"
NEW = "heap_caps_aligned_alloc(16, (needed + 15) & ~15, MALLOC_CAP_8BIT)"

TARGETS = ("conv.cpp", "depthwise_conv.cpp")


def main():
    libdeps = Path(env.subst("$PROJECT_LIBDEPS_DIR")) / env.subst("$PIOENV")  # noqa: F821
    kernels = (libdeps / "TensorFlowLite_ESP32" / "src" / "tensorflow" / "lite" /
               "micro" / "kernels")
    if not kernels.is_dir():
        print("[patch_scratch_align] library not fetched yet, nothing to patch")
        return

    for name in TARGETS:
        src = kernels / name
        if not src.is_file():
            continue
        text = src.read_text()
        if NEW in text:
            continue
        if OLD not in text:
            # depthwise_conv.cpp is patched by another hook that may not have run
            # yet on a clean tree; conv.cpp not matching means the fork changed.
            print(f"[patch_scratch_align] no unaligned allocation found in {name}")
            continue
        src.write_text(text.replace(OLD, NEW))
        print(f"[patch_scratch_align] {name}: scratch buffer now 16-byte aligned")


main()
