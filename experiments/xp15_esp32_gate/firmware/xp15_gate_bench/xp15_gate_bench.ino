// XP15 stage B — run the gate on the XIAO ESP32-S3 and check it against the answers.
//
// No camera, no SD card, no WiFi. The frames and the model live in flash, and the
// scores each frame produced off-device are compiled in beside them, so the board
// prints a verdict instead of a log somebody has to diff by eye.
//
// That isolation is the point. The question here is whether this chip runs this
// model and gets the same answers; a camera would add exposure, focus and a
// different scene distribution, all of which can make the answer wrong for reasons
// that have nothing to do with the chip. Whether a lens pointed at real sky
// produces the same false-alarm rate is a separate experiment and needs real sky.
//
// Board:   Seeed XIAO ESP32-S3 (Sense)
//
// Two ways to build this, and they differ by more than convenience:
//
//   Arduino IDE, the quick one. Boards Manager -> "esp32" by Espressif, then
//   Library Manager -> "TensorFlowLite_ESP32". Builds as written. That library
//   ships generic reference kernels, so treat its timings as an upper bound.
//
//   ESP-IDF with Espressif's own "esp-tflite-micro" component, the one worth
//   quoting. It carries the ESP-NN kernels that use the S3's vector instructions
//   and is materially faster on exactly these convolutions. The code below is the
//   same; only the include at the top and the project scaffolding change.
//
// If the number that matters is latency, use the second and say which you used --
// reporting a reference-kernel timing as this chip's capability would understate
// it by a large factor.
//
// Set:     Tools -> PSRAM -> "OPI PSRAM"     (the arena does not fit without it)
//          Tools -> Partition Scheme -> "Huge APP" (540 KB of data in flash)

#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "model_data.h"
#include "frames_data.h"

namespace {

// The arena is allocated from PSRAM at runtime, not declared as a static array.
// This chip has 512 KB of internal SRAM in total and the first layer's activations
// alone are 96*96*32 bytes, so a static arena large enough to hold them would
// either fail to link or leave nothing for the stack. The 8 MB of PSRAM is exactly
// what this board has that a plain ESP32 does not, and it is why the model fits.
//
// Sized by trial: start high, read the "arena used" line this sketch prints, then
// trim. Too small fails loudly at AllocateTensors; too large only wastes PSRAM, so
// erring high on the first flash is the cheaper mistake.
constexpr size_t kArenaSize = 500 * 1024;
uint8_t* g_arena = nullptr;

// Required by this vintage of TFLite Micro. Upstream later gave the interpreter a
// default error reporter and then removed the parameter, so a newer TFLM will
// reject this argument -- which is the sort of thing that makes a sketch look
// broken when only the library moved underneath it.
tflite::MicroErrorReporter g_error_reporter;

const tflite::Model* g_tflite_model = nullptr;
tflite::MicroInterpreter* g_interpreter = nullptr;
TfLiteTensor* g_input = nullptr;
TfLiteTensor* g_output = nullptr;

float sigmoidf(float x) { return 1.0f / (1.0f + expf(-x)); }

}  // namespace

void setup() {
  Serial.begin(115200);
  while (!Serial && millis() < 3000) { }
  Serial.println();
  Serial.println("XP15 gate benchmark");
  Serial.printf("frames %d at %dx%d, model %u bytes\n",
                N_FRAMES, FRAME_RES, FRAME_RES, g_model_len);

  g_arena = (uint8_t*)heap_caps_aligned_alloc(16, kArenaSize, MALLOC_CAP_SPIRAM);
  if (g_arena == nullptr) {
    Serial.println("FATAL: no PSRAM for the arena.");
    Serial.println("  Tools -> PSRAM must be set to \"OPI PSRAM\" for this board.");
    return;
  }
  Serial.printf("arena %u KB from PSRAM\n", (unsigned)(kArenaSize / 1024));

  g_tflite_model = tflite::GetModel(g_model);
  if (g_tflite_model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.printf("FATAL: schema %lu, expected %d\n",
                  (unsigned long)g_tflite_model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // Ops listed one by one rather than pulling in AllResolver: the full resolver
  // links every kernel TFLite Micro has and costs a large amount of flash for
  // kernels this graph never reaches. If a new op is added to the model this fails
  // at AllocateTensors naming the missing op, which is a clear enough error.
  //
  // This list is read off the converted model, not guessed:
  //   CONV_2D 6, DEPTHWISE_CONV_2D 5, PAD 3, MEAN 1, FULLY_CONNECTED 1
  // PAD is here because the Keras model pads explicitly rather than using 'same',
  // which is what keeps it numerically identical to the torch original. There is
  // no QUANTIZE op: the model takes int8 in and gives int8 out.
  static tflite::MicroMutableOpResolver<5> resolver;
  resolver.AddConv2D();
  resolver.AddDepthwiseConv2D();
  resolver.AddPad();
  resolver.AddMean();          // GlobalAveragePooling2D lowers to MEAN
  resolver.AddFullyConnected();

  static tflite::MicroInterpreter interpreter(g_tflite_model, resolver,
                                              g_arena, kArenaSize,
                                              &g_error_reporter);
  g_interpreter = &interpreter;
  if (g_interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("FATAL: AllocateTensors failed -- raise kArenaSize");
    return;
  }

  g_input = g_interpreter->input(0);
  g_output = g_interpreter->output(0);
  Serial.printf("arena used %u of %d bytes\n",
                (unsigned)g_interpreter->arena_used_bytes(), kArenaSize);
  Serial.printf("input %dx%dx%d type %d, scale %.6f zero %d\n",
                g_input->dims->data[1], g_input->dims->data[2],
                g_input->dims->data[3], g_input->type,
                g_input->params.scale, g_input->params.zero_point);
  Serial.println("frame,label,ref,board,abs_diff,us");
}

void loop() {
  if (g_interpreter == nullptr) { delay(5000); return; }

  uint32_t total_us = 0;
  float worst = 0.0f;
  int mismatches = 0;

  for (int i = 0; i < N_FRAMES; i++) {
    const unsigned char* frame = g_frames + (size_t)i * FRAME_BYTES;

    // uint8 [0,255] -> the model's int8 input. The exported scale and zero point
    // are applied here rather than baked into the header so that a requantized
    // model can be dropped in without regenerating the frames.
    for (int p = 0; p < FRAME_BYTES; p++) {
      float v = frame[p] / 255.0f;
      int32_t q = lrintf(v / g_input->params.scale) + g_input->params.zero_point;
      g_input->data.int8[p] = (int8_t)(q < -128 ? -128 : (q > 127 ? 127 : q));
    }

    uint32_t t0 = micros();
    TfLiteStatus st = g_interpreter->Invoke();
    uint32_t dt = micros() - t0;
    if (st != kTfLiteOk) { Serial.printf("%d,INVOKE_FAILED\n", i); continue; }
    total_us += dt;

    float logit = (g_output->data.int8[0] - g_output->params.zero_point) *
                  g_output->params.scale;
    float score = sigmoidf(logit);
    float diff = fabsf(score - g_ref_score[i]);
    if (diff > worst) worst = diff;
    // 1e-2 is loose on purpose: it catches a broken port, which misses by tenths,
    // without tripping on the last bit of a fixed-point rounding difference.
    if (diff > 0.01f) mismatches++;

    Serial.printf("%d,%s,%.5f,%.5f,%.5f,%lu\n", i, g_label[i], g_ref_score[i],
                  score, diff, (unsigned long)dt);
  }

  Serial.printf("\nmean %.2f ms/frame over %d frames\n",
                total_us / 1000.0f / N_FRAMES, N_FRAMES);
  Serial.printf("worst |board - reference| %.5f, %d frames over 0.01\n",
                worst, mismatches);
  Serial.println(mismatches == 0 ? "PORT OK" : "PORT MISMATCH -- investigate");
  Serial.printf("free heap %u, free PSRAM %u\n",
                (unsigned)ESP.getFreeHeap(), (unsigned)ESP.getFreePsram());

  Serial.println("\nholding; reset to run again");
  while (true) { delay(10000); }
}
