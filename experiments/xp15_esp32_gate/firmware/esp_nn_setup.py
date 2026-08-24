"""PlatformIO pre-build hook: make the fetched esp-nn buildable as an Arduino library.

esp-nn is an ESP-IDF component, not an Arduino library. Its CMakeLists names the
exact per-architecture sources to compile and sets the target macro that selects
them; PlatformIO knows none of that, generates a bare manifest, and then tries to
compile every file in the tree. That fails twice over: headers live in
subdirectories the include path does not cover, and the RISC-V assembly cannot be
assembled for an Xtensa core.

This writes the manifest the library would have had:

* **the target macro.** ``esp_nn.h`` maps every public name onto either the ANSI C
  fallback or the S3 kernels depending on ``CONFIG_IDF_TARGET_ESP32S3``. Without
  it the header compiles, links, and quietly gives you the slow path -- which is
  the failure mode worth guarding against here, because it looks like success.
* **the include directories**, which are ``include/`` plus two source folders that
  hold private headers.
* **a source filter** that drops the RISC-V and ESP32-P4 variants. The remaining
  ANSI and S3 files coexist: the header renames them apart, so both compiling
  costs some flash and changes nothing else.

Written as a hook rather than by vendoring esp-nn into this repo, because the
library is several megabytes and reproducible from its git URL.
"""

import json
from pathlib import Path

Import("env")  # noqa: F821  -- injected by PlatformIO


MANIFEST = {
    "name": "esp-nn",
    "version": "1.0.0",
    "build": {
        "includeDir": "include",
        "srcDir": "src",
        # Everything except the architectures this chip is not.
        # PlatformIO's globs do not cross directory separators, and every source
        # here sits one level down (convolution/, activation_functions/, ...), so
        # a bare "-<*riscv*>" silently matches nothing and the RISC-V assembly is
        # compiled anyway. Both depths are listed.
        "srcFilter": [
            "+<*>",
            "-<*riscv*>", "-<*/*riscv*>", "-<*/*/*riscv*>",
            "-<*esp32p4*>", "-<*/*esp32p4*>", "-<*/*/*esp32p4*>",
            "-<test*>", "-<*/test*>",
        ],
        "flags": [
            "-DCONFIG_IDF_TARGET_ESP32S3",
            "-I$PROJECT_LIBDEPS_DIR/$PIOENV/esp-nn/include",
            "-I$PROJECT_LIBDEPS_DIR/$PIOENV/esp-nn/src/common",
            "-I$PROJECT_LIBDEPS_DIR/$PIOENV/esp-nn/src/softmax",
        ],
    },
}


def main():
    libdeps = Path(env.subst("$PROJECT_LIBDEPS_DIR")) / env.subst("$PIOENV")  # noqa: F821
    esp_nn = libdeps / "esp-nn"
    if not esp_nn.is_dir():
        # First run: PlatformIO has not fetched it yet. The hook runs again on the
        # next invocation, by which point the directory exists.
        print("[esp_nn_setup] esp-nn not fetched yet, nothing to patch")
        return

    manifest = esp_nn / "library.json"
    current = json.loads(manifest.read_text()) if manifest.exists() else {}
    if current.get("build") == MANIFEST["build"]:
        return

    manifest.write_text(json.dumps(MANIFEST, indent=2) + "\n")
    print(f"[esp_nn_setup] wrote {manifest} -- ESP32-S3 kernels, RISC-V excluded")


main()
