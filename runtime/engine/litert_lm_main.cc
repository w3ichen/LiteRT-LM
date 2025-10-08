// Copyright 2025 The ODML Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// ODML pipeline to execute or benchmark LLM graph on device.
//
// The pipeline does the following
// 1) Read the corresponding parameters, weight and model file paths.
// 2) Construct a graph model with the setting.
// 3) Execute model inference and generate the output.
//
// Consider run_llm_inference_engine.sh as an example to run on android device.

#include <optional>
#include <set>
#include <string>
#include <vector>

#include "absl/base/log_severity.h"  // from @com_google_absl
#include "absl/flags/flag.h"  // from @com_google_absl
#include "absl/flags/parse.h"  // from @com_google_absl
#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/log/globals.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/numbers.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "litert/c/litert_logging.h"  // from @litert
#include "runtime/engine/litert_lm_lib.h"
#include "runtime/util/status_macros.h"

ABSL_FLAG(std::string, backend, "gpu",
          "Executor backend to use for LLM execution (cpu, gpu, etc.)");
ABSL_FLAG(std::optional<std::string>, vision_backend, std::nullopt,
          "Backend to use for the vision model (cpu or gpu). If not specified, "
          "the vision backend will be chosen based on the main backend.");
ABSL_FLAG(std::optional<std::string>, audio_backend, std::nullopt,
          "Backend to use for the audio model (cpu or gpu). If not specified, "
          "the audio backend will be chosen based on the main backend.");
ABSL_FLAG(std::string, sampler_backend, "",
          "Sampler backend to use for LLM execution (cpu, gpu, etc.). If "
          "empty, the sampler backend will be chosen for the best according to "
          "the main executor, for example, gpu for gpu main executor.");
ABSL_FLAG(std::string, model_path, "", "Model path to use for LLM execution.");
ABSL_FLAG(std::string, input_prompt,
          "What is the tallest building in the world?",
          "Input prompt to use for testing LLM execution.");
ABSL_FLAG(int, max_num_tokens, 0,
          "Maximum number of tokens or context length to use for LLM execution "
          "of a graph with dynamic context length. If 0, the maximum context "
          "length will be determined by some heuristic. On benchmark mode, it "
          "will be set to one equal to or greater than "
          "benchmark_prefill_tokens + benchmark_decode_tokens.");
ABSL_FLAG(std::vector<std::string>, prefill_batch_sizes, {},
          "A list of maximum numbers of prefill tokens processed at once. If "
          "empty, it will be the list of one entry with the length of input "
          "prompt tokens or benchmark_prefill_tokens when benchmark mode is "
          "enabled.");
ABSL_FLAG(bool, benchmark, false, "Benchmark the LLM execution.");
ABSL_FLAG(int, benchmark_prefill_tokens, 0,
          "If benchmark is true and the value is larger than 0, the benchmark "
          "will use this number to set the number of prefill tokens "
          "(regardless of the input prompt).");
ABSL_FLAG(int, benchmark_decode_tokens, 0,
          "If benchmark is true and the value is larger than 0, the benchmark "
          "will use this number to set the number of decode steps (regardless "
          "of the input prompt).");
ABSL_FLAG(bool, async, true, "Run the LLM execution asynchronously.");
ABSL_FLAG(bool, report_peak_memory_footprint, false,
          "Report peak memory footprint.");
ABSL_FLAG(bool, force_f32, false,
          "Force float 32 precision for the activation data type.");
ABSL_FLAG(bool, multi_turns, false,
          "If true, the command line will ask for multi-turns input.");
ABSL_FLAG(int, num_cpu_threads, 0,
          "If greater than 0, the number of CPU threads to use for the LLM "
          "execution with CPU backend.");
ABSL_FLAG(bool, gpu_external_tensor_mode, false,
          "If false (by default), the GPU backend will use no external tensor "
          "mode which runs slightly faster during decode. It should be set "
          "true when GPU backend doesn't support no external tensor mode, "
          "e.g. Vulkan or OpenGL.");
ABSL_FLAG(bool, configure_magic_numbers, true,
          "If true and the model contains magic numbers, present magic number "
          "configs when the model is initialized.");
ABSL_FLAG(bool, verify_magic_numbers, false,
          "If true and the model contains magic numbers and test signatures, "
          "verify magic number configs when the real dimensions that replaced "
          "magic numbers match with ones of test signatures.");
ABSL_FLAG(bool, clear_kv_cache_before_prefill, false,
          "If true, clear kv cache before the first prefill step. This may "
          "help to disclose any issues related to kv cache.");
ABSL_FLAG(int, num_logits_to_print_after_decode, 0,
          "The number of values at the beginning of logits, in the middle of "
          "logits, and at the end of logits to print after each decode step. "
          "If 0, disables printing logits.");
ABSL_FLAG(std::string, score_target_text, "", "Target text to score.");
ABSL_FLAG(std::optional<std::vector<std::string>>, image_files, std::nullopt,
          "The path to the image files that to be used for vision modality.");
ABSL_FLAG(std::optional<std::vector<std::string>>, audio_files, std::nullopt,
          "The path to the audio files that to be used for audio modality.");
ABSL_FLAG(bool, gpu_madvise_original_shared_tensors, true,
          "If true, the GPU backend will madvise the original shared tensors "
          "after use.");
ABSL_FLAG(bool, disable_cache, false, "Disable weight cache.");

namespace {

// Converts an absl::LogSeverityAtLeast to a LiteRtLogSeverity.
LiteRtLogSeverity AbslMinLogLevelToLiteRtLogSeverity(
    absl::LogSeverityAtLeast min_log_level) {
  int min_log_level_int = static_cast<int>(min_log_level);
  switch (min_log_level_int) {
    case -1:
      // ABSL does not support verbose logging, but passes through -1 as a log
      // level, which we can use to enable verbose logging in LiteRT.
      return LITERT_VERBOSE;
    case static_cast<int>(absl::LogSeverityAtLeast::kInfo):
      return LITERT_INFO;
    case static_cast<int>(absl::LogSeverityAtLeast::kWarning):
      return LITERT_WARNING;
    case static_cast<int>(absl::LogSeverityAtLeast::kError):
      return LITERT_ERROR;
    case static_cast<int>(absl::LogSeverityAtLeast::kFatal):
      return LITERT_SILENT;
    default:
      return LITERT_INFO;
  }
}

absl::StatusOr<std::set<int>> ParsePrefillBatchSizes(
    const std::vector<std::string>& prefill_batch_sizes) {
  std::set<int> parsed_prefill_batch_sizes;
  for (const auto& prefill_batch_size : prefill_batch_sizes) {
    int size;
    if (!absl::SimpleAtoi(prefill_batch_size, &size)) {
      return absl::InvalidArgumentError(
          absl::StrCat("Invalid prefill batch size: ", prefill_batch_size));
    }
    parsed_prefill_batch_sizes.insert(size);
  }
  return parsed_prefill_batch_sizes;
}

absl::Status MainHelper(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  LiteRtSetMinLoggerSeverity(
      LiteRtGetDefaultLogger(),
      AbslMinLogLevelToLiteRtLogSeverity(absl::MinLogLevel()));

  if (argc <= 1) {
    ABSL_LOG(INFO)
        << "Example usage: ./litert_lm_main --model_path=<model_path> "
           "[--input_prompt=<input_prompt>] [--backend=<cpu|gpu|npu>] "
           "[--max_num_tokens=<max_num_tokens>] "
           "[--prefill_batch_sizes=<size1>[,<size2>,...]]"
           "[--vision_backend=<cpu|gpu>] [--audio_backend=<cpu|gpu>] "
           "[--image_files=<image_path1>,<image_path2>,...] "
           "[--audio_files=<audio_path1>,<audio_path2>,...] "
           "[--sampler_backend=<cpu|gpu>] [--benchmark] "
           "[--benchmark_prefill_tokens=<num_prefill_tokens>] "
           "[--benchmark_decode_tokens=<num_decode_tokens>] "
           "[--async=<true|false>] [--force_f32=<true|false] "
           "[--report_peak_memory_footprint] [--multi_turns=<true|false>] "
           "[--num_cpu_threads=<num_cpu_threads>] "
           "[--gpu_external_tensor_mode=<true|false>] "
           "[--configure_magic_numbers=<true|false>] "
           "[--verify_magic_numbers=<true|false>] "
           "[--clear_kv_cache_before_prefill=<true|false>] "
           "[--num_logits_to_print_after_decode=<num_logits_to_print>]"
           "[--score_target_text=<target_text>]"
           "[--gpu_madvise_original_shared_tensors=<true|false>]";
    return absl::InvalidArgumentError("No arguments provided.");
  }

  litert::lm::LiteRtLmSettings settings;

  settings.backend = absl::GetFlag(FLAGS_backend);
  settings.vision_backend = absl::GetFlag(FLAGS_vision_backend);
  settings.audio_backend = absl::GetFlag(FLAGS_audio_backend);
  settings.sampler_backend = absl::GetFlag(FLAGS_sampler_backend);
  settings.model_path = absl::GetFlag(FLAGS_model_path);
  settings.input_prompt = absl::GetFlag(FLAGS_input_prompt);
  settings.max_num_tokens = absl::GetFlag(FLAGS_max_num_tokens);
  ASSIGN_OR_RETURN(
      settings.prefill_batch_sizes,
      ParsePrefillBatchSizes(absl::GetFlag(FLAGS_prefill_batch_sizes)));
  settings.image_files = absl::GetFlag(FLAGS_image_files);
  settings.audio_files = absl::GetFlag(FLAGS_audio_files);
  settings.benchmark = absl::GetFlag(FLAGS_benchmark);
  settings.benchmark_prefill_tokens =
      absl::GetFlag(FLAGS_benchmark_prefill_tokens);
  settings.benchmark_decode_tokens =
      absl::GetFlag(FLAGS_benchmark_decode_tokens);
  settings.async = absl::GetFlag(FLAGS_async);
  settings.report_peak_memory_footprint =
      absl::GetFlag(FLAGS_report_peak_memory_footprint);
  settings.force_f32 = absl::GetFlag(FLAGS_force_f32);
  settings.multi_turns = absl::GetFlag(FLAGS_multi_turns);
  settings.num_cpu_threads = absl::GetFlag(FLAGS_num_cpu_threads);
  settings.gpu_external_tensor_mode =
      absl::GetFlag(FLAGS_gpu_external_tensor_mode);
  settings.configure_magic_numbers =
      absl::GetFlag(FLAGS_configure_magic_numbers);
  settings.verify_magic_numbers = absl::GetFlag(FLAGS_verify_magic_numbers);
  settings.clear_kv_cache_before_prefill =
      absl::GetFlag(FLAGS_clear_kv_cache_before_prefill);
  settings.num_logits_to_print_after_decode =
      absl::GetFlag(FLAGS_num_logits_to_print_after_decode);
  settings.score_target_text = absl::GetFlag(FLAGS_score_target_text);
  settings.gpu_madvise_original_shared_tensors =
      absl::GetFlag(FLAGS_gpu_madvise_original_shared_tensors);
  settings.disable_cache = absl::GetFlag(FLAGS_disable_cache);

  // Adjust max_num_tokens and prefill_batch_size if not set on benchmark mode.
  if (settings.benchmark && settings.benchmark_prefill_tokens > 0) {
    if (settings.max_num_tokens == 0 && settings.benchmark_decode_tokens > 0) {
      settings.max_num_tokens =
          settings.benchmark_prefill_tokens + settings.benchmark_decode_tokens;
    }
    if (settings.prefill_batch_sizes.empty()) {
      settings.prefill_batch_sizes.insert(settings.benchmark_prefill_tokens);
    }
  }

  return litert::lm::RunLiteRtLm(settings);
}

}  // namespace

int main(int argc, char** argv) {
  ABSL_CHECK_OK(MainHelper(argc, argv));
  return 0;
}
