"""PlatformIO pre-build hook: route int8 *depthwise* convolution to ESP-NN too.

The fork this project builds against patches ``conv.cpp``. That took this model
from 20.5 s per frame to 4.5 s -- a real gain, and far short of the ~100x the same
patch achieves on a three-layer network. The difference is what the model is made
of: eleven convolutions, of which **five are depthwise**, and depthwise_conv.cpp
was left on the reference C path. Those five sit at the widest activations, so
they plausibly hold most of the remaining time.

Rather than ask for a second fork, this rewrites the kernel in the fetched library
the same way the first patch did, keeping the reference implementation in an
``#else`` so the change is legible and reversible.

**The one real difference from the conv case** is the channel multiplier. TFLite's
depthwise op has a ``depth_multiplier``, and ESP-NN expects it as ``ch_mult`` with
``in_ch * ch_mult == out_ch``. This network only ever uses 1, but a model that used
more would silently compute the wrong thing if it were assumed, so it is read from
the tensor shapes and checked.

Idempotent: it looks for its own marker and does nothing if already applied.
"""

from pathlib import Path

Import("env")  # noqa: F821  -- injected by PlatformIO

MARKER = "USE_ESP_NN_DEPTHWISE"

INCLUDES = '''#ifdef USE_ESP_NN_DEPTHWISE
#include "esp_nn.h"
#include "esp_nn_defs.h"
#include <esp_heap_caps.h>
#include <cstdio>
#endif

namespace tflite {'''

FAST_PATH = '''    case kTfLiteInt8: {
#ifdef USE_ESP_NN_DEPTHWISE
      const auto& in_shape = tflite::micro::GetTensorShape(input);
      const auto& fl_shape = tflite::micro::GetTensorShape(filter);
      const auto& out_shape = tflite::micro::GetTensorShape(output);

      const int in_h = in_shape.Dims(1), in_w = in_shape.Dims(2);
      const int in_ch = in_shape.Dims(3);
      const int fl_h = fl_shape.Dims(1), fl_w = fl_shape.Dims(2);
      const int out_h = out_shape.Dims(1), out_w = out_shape.Dims(2);
      const int out_ch = out_shape.Dims(3);

      // in_ch * ch_mult == out_ch by definition of the op. Reading it from the
      // shapes rather than trusting params.depth_multiplier means a model that
      // disagrees falls back to the reference kernel instead of computing
      // something wrong quickly.
      const int ch_mult = (in_ch > 0) ? (out_ch / in_ch) : 0;
      if (ch_mult <= 0 || in_ch * ch_mult != out_ch) {
        goto esp_nn_depthwise_fallback;
      }

      {
        data_dims_t in_dims = {in_w, in_h, in_ch, 1};
        data_dims_t fl_dims = {fl_w, fl_h, 0, 0};
        data_dims_t out_dims = {out_w, out_h, out_ch, 1};

        const DepthwiseParams dw_tf = DepthwiseConvParamsQuantized(params, data);

        dw_conv_params_t dw_params;
        dw_params.in_offset = dw_tf.input_offset;
        dw_params.out_offset = dw_tf.output_offset;
        dw_params.ch_mult = ch_mult;
        dw_params.stride.width = params.stride_width;
        dw_params.stride.height = params.stride_height;
        dw_params.padding.width = dw_tf.padding_values.width;
        dw_params.padding.height = dw_tf.padding_values.height;
        dw_params.dilation.width = params.dilation_width_factor;
        dw_params.dilation.height = params.dilation_height_factor;
        dw_params.activation.min = dw_tf.quantized_activation_min;
        dw_params.activation.max = dw_tf.quantized_activation_max;

        quant_data_t quant_data;
        quant_data.mult = const_cast<int32_t*>(data.per_channel_output_multiplier);
        quant_data.shift = const_cast<int32_t*>(data.per_channel_output_shift);

        static int8_t* dw_scratch = nullptr;
        static int dw_scratch_size = 0;
        int needed = esp_nn_get_depthwise_conv_scratch_size(
            &in_dims, &fl_dims, &out_dims, &dw_params);
        if (needed > dw_scratch_size) {
          if (dw_scratch) heap_caps_free(dw_scratch);
          // 16-byte aligned: the S3 kernels load 128 bits at a time and round a
          // misaligned base down, which lands on the allocator's block header.
          dw_scratch = (int8_t*)heap_caps_aligned_alloc(
              16, (needed + 15) & ~15, MALLOC_CAP_8BIT);
          dw_scratch_size = dw_scratch ? needed : 0;
        }
        esp_nn_set_depthwise_conv_scratch_buf(dw_scratch);

        esp_nn_depthwise_conv_s8(
            &in_dims, tflite::micro::GetTensorData<int8_t>(input),
            &fl_dims, tflite::micro::GetTensorData<int8_t>(filter),
            tflite::micro::GetOptionalTensorData<int32_t>(bias),
            &out_dims, tflite::micro::GetTensorData<int8_t>(output),
            &dw_params, &quant_data);
        break;
      }
    esp_nn_depthwise_fallback:
#endif
'''


def main():
    libdeps = Path(env.subst("$PROJECT_LIBDEPS_DIR")) / env.subst("$PIOENV")  # noqa: F821
    src = (libdeps / "TensorFlowLite_ESP32" / "src" / "tensorflow" / "lite" /
           "micro" / "kernels" / "depthwise_conv.cpp")
    if not src.is_file():
        print("[patch_depthwise] library not fetched yet, nothing to patch")
        return

    text = src.read_text()
    if MARKER in text:
        # Already patched -- but possibly by an older version of this file. The
        # first one allocated the scratch buffer with heap_caps_malloc, which is
        # only 8-byte aligned and corrupts the heap under the S3 kernels. A stale
        # patch is invisible to the marker check and survives a rebuild, so it is
        # named here rather than left to be rediscovered on the board.
        if "heap_caps_malloc(needed" in text:
            print("[patch_depthwise] WARNING: depthwise_conv.cpp carries an older "
                  "patch with an unaligned scratch buffer. Run "
                  "`rm -rf .pio/libdeps` and rebuild, or patch_scratch_align.py "
                  "will fix the allocation in place.")
        return

    if "namespace tflite {" not in text or "case kTfLiteInt8: {" not in text:
        print("[patch_depthwise] depthwise_conv.cpp does not look as expected, "
              "leaving it alone -- the library version has probably moved")
        return

    text = text.replace("namespace tflite {", INCLUDES, 1)
    text = text.replace("    case kTfLiteInt8: {", FAST_PATH, 1)
    src.write_text(text)
    print(f"[patch_depthwise] patched {src.name} -- depthwise int8 now goes to ESP-NN")


main()
