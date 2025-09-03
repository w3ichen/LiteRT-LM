#include "runtime/engine/litert_lm_lib.h"

#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "runtime/engine/engine.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/engine/io_types.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/llm_executor_settings.h"
#include "runtime/util/status_macros.h"  // IWYU pragma: keep
#include "tflite/profiling/memory_usage_monitor.h"  // from @litert

namespace litert {
namespace lm {

using ::litert::lm::Backend;
using ::litert::lm::Engine;
using ::litert::lm::EngineSettings;
using ::litert::lm::InferenceObservable;
using ::litert::lm::InputData;
using ::litert::lm::InputText;
using ::litert::lm::LlmExecutorSettings;
using ::litert::lm::ModelAssets;

// Memory check interval in milliseconds.
constexpr int kMemoryCheckIntervalMs = 50;
// Timeout duration for waiting until the engine is done with all the tasks.
const absl::Duration kWaitUntilDoneTimeout = absl::Minutes(10);

namespace {

void RunBenchmark(const LiteRtLmSettings& settings, litert::lm::Engine* llm,
                  litert::lm::Engine::Session* session) {
  const bool is_dummy_input = settings.benchmark_prefill_tokens > 0 ||
                              settings.benchmark_decode_tokens > 0;
  std::string input_prompt = settings.input_prompt;

  std::vector<litert::lm::InputData> inputs;
  inputs.emplace_back(InputText(input_prompt));

  if (settings.async) {
    if (is_dummy_input) {
      ABSL_LOG(FATAL) << "Async mode does not support benchmarking with "
                         "specified number of prefill or decode tokens. If you "
                         "want to benchmark the model, please try again with "
                         "async=false.";
    }
    InferenceObservable observable;
    absl::Status status = session->GenerateContentStream(inputs, &observable);
    ABSL_CHECK_OK(status);
    ABSL_CHECK_OK(llm->WaitUntilDone(kWaitUntilDoneTimeout));
  } else {
    auto responses = session->GenerateContent(inputs);
    ABSL_CHECK_OK(responses);
    if (!is_dummy_input) {
      ABSL_LOG(INFO) << "Responses: " << *responses;
    }
  }

  auto benchmark_info = session->GetBenchmarkInfo();
  ABSL_LOG(INFO) << *benchmark_info;
}

void RunSingleTurn(const LiteRtLmSettings& settings, litert::lm::Engine* llm,
                   litert::lm::Engine::Session* session,
                   std::string& input_prompt) {
  std::vector<litert::lm::InputData> inputs;
  inputs.emplace_back(InputText(input_prompt));
  if (settings.async) {
    InferenceObservable observable;
    absl::Status status = session->GenerateContentStream(inputs, &observable);
    ABSL_CHECK_OK(status);
    ABSL_CHECK_OK(llm->WaitUntilDone(kWaitUntilDoneTimeout));
  } else {
    auto responses = session->GenerateContent(inputs);
    ABSL_CHECK_OK(responses);
    ABSL_LOG(INFO) << "Responses: " << *responses;
  }
}

void RunMultiTurnConversation(const LiteRtLmSettings& settings,
                              litert::lm::Engine* llm,
                              litert::lm::Engine::Session* session) {
  if (settings.benchmark) {
    ABSL_LOG(FATAL) << "Benchmarking with multi-turns input is not supported.";
  }

  std::string input_prompt;
  do {
    std::cout << "Please enter the prompt (or press Enter to end): ";
    std::getline(std::cin, input_prompt);
    if (input_prompt.empty()) {
      break;
    }
    RunSingleTurn(settings, llm, session, input_prompt);
  } while (true);
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
  ASSIGN_OR_RETURN(
      EngineSettings engine_settings,
      EngineSettings::CreateDefault(std::move(model_assets), backend));
  if (settings.force_f32) {
    engine_settings.GetMutableMainExecutorSettings().SetActivationDataType(
        litert::lm::ActivationDataType::FLOAT32);
  }
  if (backend == Backend::CPU && settings.num_cpu_threads > 0) {
    auto& executor_settings = engine_settings.GetMutableMainExecutorSettings();
    ASSIGN_OR_RETURN(
        auto cpu_settings,
        executor_settings.MutableBackendConfig<litert::lm::CpuConfig>());
    cpu_settings.number_of_threads = settings.num_cpu_threads;
    executor_settings.SetBackendConfig(cpu_settings);
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
    }
  }
  ABSL_LOG(INFO) << "executor_settings: "
                 << engine_settings.GetMainExecutorSettings();

  if (settings.benchmark) {
    litert::lm::proto::BenchmarkParams benchmark_params;
    benchmark_params.set_num_prefill_tokens(settings.benchmark_prefill_tokens);
    benchmark_params.set_num_decode_tokens(settings.benchmark_decode_tokens);
    engine_settings.GetMutableBenchmarkParams() = benchmark_params;
  }
  ABSL_LOG(INFO) << "Creating engine";
  absl::StatusOr<std::unique_ptr<litert::lm::Engine>> llm =
      litert::lm::Engine::CreateEngine(std::move(engine_settings));
  ABSL_CHECK_OK(llm) << "Failed to create engine";

  ABSL_LOG(INFO) << "Creating session";
  absl::StatusOr<std::unique_ptr<litert::lm::Engine::Session>> session =
      (*llm)->CreateSession(session_config);
  ABSL_CHECK_OK(session) << "Failed to create session";

  if (settings.benchmark) {
    RunBenchmark(settings, llm->get(), session->get());
  } else if (settings.multi_turns) {
    RunMultiTurnConversation(settings, llm->get(), session->get());
  } else {
    std::string input_prompt = settings.input_prompt;
    RunSingleTurn(settings, llm->get(), session->get(), input_prompt);
  }

  if (settings.report_peak_memory_footprint) {
    float peak_mem_mb = 0.0f;
    if (mem_monitor != nullptr) {
      mem_monitor->Stop();
      peak_mem_mb = mem_monitor->GetPeakMemUsageInMB();
    }
    ABSL_LOG(INFO) << "Peak system ram usage: " << peak_mem_mb << "MB.";
  }
  return absl::OkStatus();
}

}  // namespace lm
}  // namespace litert
