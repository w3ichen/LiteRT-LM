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

#include "runtime/engine/litert_lm_lib.h"

#include <cstdint>
#include <filesystem>  // NOLINT
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "runtime/engine/engine.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/engine/io_types.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/llm_executor_settings.h"
#include "runtime/util/status_macros.h"  // IWYU pragma: keep
#include "re2/re2.h"  // from @com_googlesource_code_re2
#include "tflite/profiling/memory_info.h"  // from @litert
#include "tflite/profiling/memory_usage_monitor.h"  // from @litert

namespace litert {
namespace lm {

using ::litert::lm::Backend;
using ::litert::lm::Engine;
using ::litert::lm::EngineSettings;
using ::litert::lm::InferenceCallbacks;
using ::litert::lm::InputAudio;
using ::litert::lm::InputData;
using ::litert::lm::InputImage;
using ::litert::lm::InputText;
using ::litert::lm::LlmExecutorSettings;
using ::litert::lm::ModelAssets;

// Memory check interval in milliseconds.
constexpr int kMemoryCheckIntervalMs = 50;
// Timeout duration for waiting until the engine is done with all the tasks.
const absl::Duration kWaitUntilDoneTimeout = absl::Minutes(10);

namespace {

class LiteRtLmLibCallbacks : public InferenceCallbacks {
 public:
  explicit LiteRtLmLibCallbacks(bool is_dummy_io) {
    is_dummy_io_ = is_dummy_io;
  }

  void OnNext(const Responses& responses) override {
    if (!is_dummy_io_) {
      std::cout << *responses.GetResponseTextAt(0) << std::flush;
    }
  }

 private:
  bool is_dummy_io_;
};

void RunSingleTurn(const LiteRtLmSettings& settings, litert::lm::Engine* engine,
                   litert::lm::Engine::Session* session,
                   std::string& input_prompt,
                   std::vector<std::string>& images_bytes,
                   std::vector<std::string>& audio_bytes) {
  std::vector<litert::lm::InputData> inputs;
  auto image_input_it = images_bytes.begin();
  auto audio_input_it = audio_bytes.begin();
  RE2 re_delimiter("(<start_of_audio>|<start_of_image>)");
  absl::string_view prompt_view(input_prompt);
  const char* start = prompt_view.data();
  std::string part;
  while (RE2::FindAndConsume(&prompt_view, re_delimiter, &part)) {
    absl::string_view text_part(start, prompt_view.data() - part.size());
    if (!text_part.empty()) {
      inputs.push_back(InputText(std::string(text_part)));
    }
    start = prompt_view.data();
    if (part == "<start_of_image>") {
      inputs.emplace_back(InputImage(*image_input_it));
      ++image_input_it;
    } else if (part == "<start_of_audio>") {
      inputs.emplace_back(InputAudio(*audio_input_it));
      ++audio_input_it;
    }
  }
  if (!prompt_view.empty()) {
    inputs.push_back(InputText(std::string(prompt_view)));
  }
  if (image_input_it != images_bytes.end()) {
    ABSL_LOG(FATAL) << "The number of images is not the same as the number of "
                       "<start_of_image> tags in the prompt.";
  }
  if (audio_input_it != audio_bytes.end()) {
    ABSL_LOG(FATAL) << "The number of audio is not the same as the number of "
                       "<start_of_audio> tags in the prompt.";
  }

  bool is_dummy_io = settings.benchmark_prefill_tokens > 0 ||
                     settings.benchmark_decode_tokens > 0;
  if (is_dummy_io) {
    ABSL_LOG(INFO) << "Streaming response are not shown because dummy input "
                      "or output are used.";
  }

  if (settings.async) {
    absl::Status status = session->GenerateContentStream(
        inputs, std::make_unique<LiteRtLmLibCallbacks>(is_dummy_io));
    ABSL_CHECK_OK(status);
    ABSL_CHECK_OK(engine->WaitUntilDone(kWaitUntilDoneTimeout));
  } else {
    auto responses = session->GenerateContent(inputs);
    ABSL_CHECK_OK(responses);
    if (!is_dummy_io) {
      std::cout << "Responses: " << *responses << std::endl;
    }
  }

  if (settings.benchmark) {
    auto benchmark_info = session->GetBenchmarkInfo();
    ABSL_LOG(INFO) << *benchmark_info;
  }
}

void RunMultiTurnConversation(const LiteRtLmSettings& settings,
                              litert::lm::Engine* engine,
                              litert::lm::Engine::Session* session) {
  std::string input_prompt;
  do {
    std::cout << "Please enter the prompt (or press Enter to end): ";
    std::getline(std::cin, input_prompt);
    if (input_prompt.empty()) {
      break;
    }
    std::vector<std::string> image_bytes;
    std::vector<std::string> audio_bytes;
    RunSingleTurn(settings, engine, session, input_prompt, image_bytes,
                  audio_bytes);
  } while (true);
}

void RunScoreText(litert::lm::Engine* llm, litert::lm::Engine::Session* session,
                  std::string& input_prompt, std::string& target_text) {
  std::vector<litert::lm::InputData> inputs;
  inputs.emplace_back(InputText(input_prompt));
  std::vector<absl::string_view> target_text_vector;
  target_text_vector.push_back(target_text);
  ABSL_CHECK_OK(session->RunPrefill(inputs));
  auto response = session->RunTextScoring(target_text_vector);
  ABSL_CHECK_OK(response);
  ABSL_LOG(INFO) << "Score: " << -1 * (*response->GetScoreAt(0)) << std::endl;
}

}  // namespace

absl::Status RunLiteRtLm(const LiteRtLmSettings& settings) {
  const std::string model_path = settings.model_path;
  if (model_path.empty()) {
    return absl::InvalidArgumentError("Model path is empty.");
  }

  std::unique_ptr<tflite::profiling::memory::MemoryUsageMonitor> mem_monitor;
  if (settings.report_peak_memory_footprint) {
    mem_monitor =
        std::make_unique<tflite::profiling::memory::MemoryUsageMonitor>(
            kMemoryCheckIntervalMs);
    mem_monitor->Start();
  }
  ABSL_LOG(INFO) << "Model path: " << model_path;
  ASSIGN_OR_RETURN(ModelAssets model_assets,  // NOLINT
                   ModelAssets::Create(model_path));
  auto backend_str = settings.backend;
  ABSL_LOG(INFO) << "Choose backend: " << backend_str;
  ASSIGN_OR_RETURN(Backend backend,
                   litert::lm::GetBackendFromString(backend_str));
  std::optional<Backend> vision_backend = std::nullopt;
  if (settings.image_files.has_value()) {
    ABSL_LOG(INFO) << "Image files are provided, setting vision backend.";
    if (settings.vision_backend.has_value()) {
      ABSL_LOG(INFO) << "Provided vision backend: " << *settings.vision_backend;
      ASSIGN_OR_RETURN(vision_backend, litert::lm::GetBackendFromString(
                                           *settings.vision_backend));
    } else {
      ABSL_LOG(INFO) << "Setting vision backend based on the main backend: "
                     << backend_str;
      vision_backend = backend;
    }
  }
  std::optional<Backend> audio_backend = std::nullopt;
  if (settings.audio_files.has_value()) {
    ABSL_LOG(INFO) << "Audio files are provided, setting audio backend.";
    if (settings.audio_backend.has_value()) {
      ABSL_LOG(INFO) << "Provided audio backend: " << *settings.audio_backend;
      ASSIGN_OR_RETURN(audio_backend, litert::lm::GetBackendFromString(
                                          *settings.audio_backend));
    } else {
      ABSL_LOG(INFO) << "Setting audio backend based on the main backend: "
                     << backend_str;
      audio_backend = backend;
    }
  }
  ASSIGN_OR_RETURN(
      EngineSettings engine_settings,
      EngineSettings::CreateDefault(std::move(model_assets), backend,
                                    vision_backend, audio_backend));
  if (settings.max_num_tokens > 0) {
    engine_settings.GetMutableMainExecutorSettings().SetMaxNumTokens(
        settings.max_num_tokens);
  }
  if (settings.force_f32) {
    engine_settings.GetMutableMainExecutorSettings().SetActivationDataType(
        litert::lm::ActivationDataType::FLOAT32);
  }
  if (settings.disable_cache) {
    engine_settings.GetMutableMainExecutorSettings().SetCacheDir(":nocache");
  }
  if (backend == Backend::CPU && settings.num_cpu_threads > 0) {
    auto& executor_settings = engine_settings.GetMutableMainExecutorSettings();
    ASSIGN_OR_RETURN(
        auto cpu_settings,
        executor_settings.MutableBackendConfig<litert::lm::CpuConfig>());
    cpu_settings.number_of_threads = settings.num_cpu_threads;
    executor_settings.SetBackendConfig(cpu_settings);
  }
  if (backend == Backend::GPU) {
    auto& executor_settings = engine_settings.GetMutableMainExecutorSettings();
    ASSIGN_OR_RETURN(
        auto gpu_settings,
        executor_settings.MutableBackendConfig<litert::lm::GpuConfig>());
    gpu_settings.external_tensor_mode = settings.gpu_external_tensor_mode;
    executor_settings.SetBackendConfig(gpu_settings);
  }
  auto session_config = litert::lm::SessionConfig::CreateDefault();
  auto sampler_backend_str = settings.sampler_backend;
  if (!sampler_backend_str.empty()) {
    auto sampler_backend =
        litert::lm::GetBackendFromString(settings.sampler_backend);
    if (!sampler_backend.ok()) {
      ABSL_LOG(WARNING) << "Ignore invalid sampler backend string: "
                        << sampler_backend.status();
    } else {
      session_config.SetSamplerBackend(*sampler_backend);
      auto& executor_settings =
          engine_settings.GetMutableMainExecutorSettings();
      executor_settings.SetSamplerBackend(*sampler_backend);
    }
  }

  AdvancedSettings advanced_settings{
      .prefill_batch_sizes = settings.prefill_batch_sizes,
      .configure_magic_numbers = settings.configure_magic_numbers,
      .verify_magic_numbers = settings.verify_magic_numbers,
      .clear_kv_cache_before_prefill = settings.clear_kv_cache_before_prefill,
      .num_logits_to_print_after_decode =
          static_cast<uint32_t>(settings.num_logits_to_print_after_decode),
      .gpu_madvise_original_shared_tensors =
          settings.gpu_madvise_original_shared_tensors,
  };
  if (advanced_settings != AdvancedSettings()) {
    engine_settings.GetMutableMainExecutorSettings().SetAdvancedSettings(
        advanced_settings);
  }

  ABSL_LOG(INFO) << "executor_settings: "
                 << engine_settings.GetMainExecutorSettings();

  if (engine_settings.GetVisionExecutorSettings().has_value()) {
    ABSL_LOG(INFO) << "vision_executor_settings: "
                   << engine_settings.GetVisionExecutorSettings().value();
  } else {
    ABSL_LOG(INFO) << "vision_executor_settings: not set";
  }
  if (engine_settings.GetAudioExecutorSettings().has_value()) {
    ABSL_LOG(INFO) << "audio_executor_settings: "
                   << engine_settings.GetAudioExecutorSettings().value();
  } else {
    ABSL_LOG(INFO) << "audio_executor_settings: not set";
  }

  if (settings.benchmark) {
    if (settings.multi_turns) {
      ABSL_LOG(FATAL)
          << "Benchmarking with multi-turns input is not supported.";
    }

    litert::lm::proto::BenchmarkParams benchmark_params;
    benchmark_params.set_num_prefill_tokens(settings.benchmark_prefill_tokens);
    benchmark_params.set_num_decode_tokens(settings.benchmark_decode_tokens);
    engine_settings.GetMutableBenchmarkParams() = benchmark_params;
  }

  ABSL_LOG(INFO) << "Creating engine";
  absl::StatusOr<std::unique_ptr<litert::lm::Engine>> engine =
      litert::lm::Engine::CreateEngine(std::move(engine_settings),
                                       settings.input_prompt);
  ABSL_CHECK_OK(engine) << "Failed to create engine";

  ABSL_LOG(INFO) << "Creating session";
  absl::StatusOr<std::unique_ptr<litert::lm::Engine::Session>> session =
      (*engine)->CreateSession(session_config);
  ABSL_CHECK_OK(session) << "Failed to create session";

  if (settings.score_target_text.has_value() &&
      !settings.score_target_text->empty()) {
    std::string input_prompt = settings.input_prompt;
    std::string score_target_text = settings.score_target_text.value();
    RunScoreText(engine->get(), session->get(), input_prompt,
                 score_target_text);
  } else if (settings.multi_turns) {
    RunMultiTurnConversation(settings, engine->get(), session->get());
  } else {
    std::string input_prompt = settings.input_prompt;
    std::vector<std::string> images_bytes;

    if (settings.image_files.has_value() && !settings.image_files->empty()) {
      for (const auto& image_file : *settings.image_files) {
        ABSL_LOG(INFO) << "Loading image from: " << image_file;
        std::ifstream file_stream(image_file, std::ios::binary);
        if (!file_stream) {
          return absl::InternalError(
              absl::StrCat("Failed to open image file: ", image_file));
        }
        std::stringstream buffer;
        buffer << file_stream.rdbuf();
        images_bytes.push_back(buffer.str());
      }
    }
    std::vector<std::string> audio_bytes;
    if (settings.audio_files.has_value() && !settings.audio_files->empty()) {
      for (const auto& audio_file : *settings.audio_files) {
        ABSL_LOG(INFO) << "Loading audio from: " << audio_file;
        std::ifstream file_stream(audio_file, std::ios::binary);
        if (!file_stream) {
          return absl::InternalError(
              absl::StrCat("Failed to open audio file: ", audio_file));
        }
        std::stringstream buffer;
        buffer << file_stream.rdbuf();
        audio_bytes.push_back(buffer.str());
      }
    }
    RunSingleTurn(settings, engine->get(), session->get(), input_prompt,
                  images_bytes, audio_bytes);
  }

  // Manually releasing the session to ensure that memory usage from
  // `GetMemoryUsage()` is reporting idle engine state without active sessions.
  session->release();

  if (settings.report_peak_memory_footprint) {
    float peak_mem_mb = 0.0f;
    if (mem_monitor != nullptr) {
      mem_monitor->Stop();
      peak_mem_mb = mem_monitor->GetPeakMemUsageInMB();
    }
    ABSL_LOG(INFO) << "Peak system ram usage: " << peak_mem_mb << "MB.";
    ABSL_LOG(INFO) << "Memory usage: "
                   << tflite::profiling::memory::GetMemoryUsage();
  }

  return absl::OkStatus();
}

}  // namespace lm
}  // namespace litert
