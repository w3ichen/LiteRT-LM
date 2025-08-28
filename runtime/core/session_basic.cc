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

#include "runtime/core/session_basic.h"

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "runtime/components/sampler.h"
#include "runtime/components/sampler_factory.h"
#include "runtime/components/stop_token_detector.h"
#include "runtime/components/tokenizer.h"
#include "runtime/core/pipeline.h"
#include "runtime/engine/engine.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/engine/io_types.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/llm_executor.h"
#include "runtime/executor/llm_executor_io_types.h"
#include "runtime/framework/threadpool.h"
#include "runtime/proto/sampler_params.pb.h"
#include "runtime/util/convert_tensor_buffer.h"
#include "runtime/util/litert_status_util.h"
#include "runtime/util/status_macros.h"  // IWYU pragma: keep

namespace litert::lm {

// static
absl::StatusOr<std::unique_ptr<SessionBasic>> SessionBasic::Create(
    LlmExecutor* executor, Tokenizer* tokenizer,
    const SessionConfig& session_config,
    std::optional<BenchmarkInfo> benchmark_info,
    ThreadPool* worker_thread_pool) {
  auto sampler_backend = session_config.GetSamplerBackend();
  std::unique_ptr<Sampler> sampler;
  // If use CPU sampling, we create it here; For GPU sampling, we let executor
  // create it internally.
  if (sampler_backend == Backend::CPU) {
    ASSIGN_OR_RETURN(
        sampler,
        CreateSampler(sampler_backend, session_config.GetNumOutputCandidates(),
                      session_config.GetSamplerParams()));
  } else if (sampler_backend != Backend::GPU &&
             sampler_backend != Backend::NPU) {
    return absl::InvalidArgumentError(
        absl::StrCat("Unsupported sampler backend: ", sampler_backend));
  }

  if (benchmark_info.has_value()) {
    ABSL_LOG(INFO) << "Benchmark is enabled.";
  }
  StopTokenDetector stop_token_detector(
      session_config.GetNumOutputCandidates());
  for (const auto& stop_token_sequence : session_config.GetStopTokenIds()) {
    RETURN_IF_ERROR(
        stop_token_detector.AddStopTokenSequence(stop_token_sequence));
  }
  return absl::WrapUnique(new SessionBasic(
      executor, tokenizer, std::move(sampler), session_config, benchmark_info,
      worker_thread_pool, stop_token_detector));
}

SessionBasic::~SessionBasic() {
  auto status = executor_.Reset();
  if (!status.ok()) {
    ABSL_LOG(ERROR) << "Failed to reset executor: " << status;
  }
}

absl::StatusOr<std::string> SessionBasic::ApplyPromptTemplates(
    absl::string_view input) {
  return absl::StrCat(session_config_.GetPromptTemplates().user().prefix(),
                      input,
                      session_config_.GetPromptTemplates().user().suffix(),
                      session_config_.GetPromptTemplates().model().prefix());
}

// TODO - b/436674053: Modulize the preprocessing logic into a separate
// preprocessor class, please refer to the bug for more details.
absl::StatusOr<std::vector<InputData>> SessionBasic::PreprocessContents(
    const std::vector<InputData>& contents) {
  std::vector<InputData> preprocessed_contents;
  for (const auto& input : contents) {
    if (const auto* input_text = std::get_if<InputText>(&input)) {
      if (input_text->IsTensorBuffer()) {
        ASSIGN_OR_RETURN(const auto* token_ids,
                         input_text->GetPreprocessedTextTensor());
        LITERT_ASSIGN_OR_RETURN_ABSL(auto token_ids_clone,
                                     token_ids->Duplicate());
        preprocessed_contents.emplace_back(
            InputText(std::move(token_ids_clone)));
      } else {
        ASSIGN_OR_RETURN(auto raw_text, input_text->GetRawTextString());
        ASSIGN_OR_RETURN(auto formatted_text, ApplyPromptTemplates(raw_text));
        int benchmark_prefill_token_count = 0;
        if (benchmark_info_.has_value()) {
          benchmark_prefill_token_count =
              benchmark_info_->GetBenchmarkParams().num_prefill_tokens();
          RETURN_IF_ERROR(benchmark_info_->TimePrefillTurnStart());
        }
        ASSIGN_OR_RETURN(std::vector<int> ids,
                         tokenizer_.TextToTokenIds(formatted_text));
        if (benchmark_prefill_token_count > 0) {
          // If benchmark is enabled, we will use the benchmark prefill token
          // count to set the prefill token count.
          ids.resize(benchmark_prefill_token_count);
        } else {
          // TODO(hoko): Ask @ztenghui what is the original design intent here.
          ids.insert(ids.begin(), session_config_.GetStartTokenId());
        }
        ASSIGN_OR_RETURN(auto ids_buffer,
                         tokenizer_.TokenIdsToTensorBuffer(ids));
        preprocessed_contents.emplace_back(InputText(std::move(ids_buffer)));
      }
    } else if (const auto* input_image = std::get_if<InputImage>(&input)) {
      return absl::UnimplementedError("Image prefill is not implemented yet.");
    } else if (const auto* input_audio = std::get_if<InputAudio>(&input)) {
      return absl::UnimplementedError("Audio prefill is not implemented yet.");
    }
  }
  return preprocessed_contents;
}

absl::Status SessionBasic::PrefillInternal(
    const std::vector<InputData>& preprocessed_contents,
    bool wait_for_completion) {
  // TODO(b/397975034): Consider to utilize a prompt formatting logic in a
  // separate library/class.
  // Update the input with prompt formatting.
  RET_CHECK(preprocessed_contents.size() == 1)
      << "preprocessed_contents must have exactly one element.";
  RET_CHECK(std::holds_alternative<InputText>(preprocessed_contents.at(0)))
      << "preprocessed_contents must have an InputText.";
  const InputText& input_text =
      std::get<InputText>(preprocessed_contents.at(0));
  ASSIGN_OR_RETURN(const auto* token_ids,
                   input_text.GetPreprocessedTextTensor());
  LITERT_ASSIGN_OR_RETURN_ABSL(auto token_ids_clone, token_ids->Duplicate());
  ExecutorInputs inputs(ExecutorTextData(std::move(token_ids_clone)),
                        std::nullopt, std::nullopt);
  // This should be added to the beginning of the next prefill call as will no?
  // Also, this is not thread safe. More discussion with @ztenghui is needed.
  ASSIGN_OR_RETURN(
      last_prefill_token_id_,
      Prefill(executor_, inputs, wait_for_completion, benchmark_info_));
  return absl::OkStatus();
}

absl::Status SessionBasic::RunPrefill(const std::vector<InputData>& contents) {
  if (contents.empty()) {
    return absl::InvalidArgumentError("Input is empty.");
  }
  ASSIGN_OR_RETURN(std::vector<InputData> preprocessed_contents,
                   PreprocessContents(contents));
  absl::Status status;
  RETURN_IF_ERROR(worker_thread_pool_.Schedule(
      [this, preprocessed_contents = std::move(preprocessed_contents),
       &status]() {
        status = this->PrefillInternal(preprocessed_contents,
                                       /*wait_for_completion=*/true);
      }));
  RETURN_IF_ERROR(worker_thread_pool_.WaitUntilDone(Engine::kDefaultTimeout));
  return status;
}

absl::Status SessionBasic::RunPrefillAsync(
    const std::vector<InputData>& contents, InferenceObservable* observer) {
  if (contents.empty()) {
    return absl::InvalidArgumentError("Input is empty.");
  }
  ASSIGN_OR_RETURN(std::vector<InputData> preprocessed_contents,
                   PreprocessContents(contents));
  RETURN_IF_ERROR(worker_thread_pool_.Schedule(
      [this, preprocessed_contents = std::move(preprocessed_contents),
       observer]() {
        absl::Status status = this->PrefillInternal(
            preprocessed_contents, /*wait_for_completion=*/false);
        ABSL_LOG(INFO) << "RunPrefillAsync status: " << status;
        if (status.ok()) {
          observer->OnDone();
        } else {
          observer->OnError(status);
        }
      }));
  return absl::OkStatus();
}

absl::StatusOr<Responses> SessionBasic::DecodeInternal() {
  if (sampler_ == nullptr) {
    ASSIGN_OR_RETURN(
        auto responses,
        Decode(executor_, tokenizer_, stop_token_detector_, benchmark_info_));
    return responses;
  } else {
    std::vector<int> decoded_ids(session_config_.GetNumOutputCandidates(),
                                 last_prefill_token_id_);
    auto decoded_ids_buffer = CopyToTensorBuffer<int>(
        decoded_ids, {session_config_.GetNumOutputCandidates(), 1});
    ASSIGN_OR_RETURN(
        auto responses,
        DecodeCustomSampling(executor_, tokenizer_, stop_token_detector_,
                             /*num_output_candidates=*/1, *sampler_,
                             *decoded_ids_buffer, benchmark_info_));
    return responses;
  }
}

absl::Status SessionBasic::DecodeInternalStreaming(
    InferenceObservable* observer) {
  if (sampler_ == nullptr) {
    RETURN_IF_ERROR(DecodeStreaming(executor_, tokenizer_, stop_token_detector_,
                                    benchmark_info_, observer));
  } else {
    std::vector<int> decoded_ids(session_config_.GetNumOutputCandidates(),
                                 last_prefill_token_id_);
    auto decoded_ids_buffer = CopyToTensorBuffer<int>(
        decoded_ids, {session_config_.GetNumOutputCandidates(), 1});
    RETURN_IF_ERROR(DecodeCustomSamplingStreaming(
        executor_, tokenizer_, stop_token_detector_,
        /*num_output_candidates=*/1, *sampler_, *decoded_ids_buffer,
        benchmark_info_, observer));
  }
  return absl::OkStatus();
}

absl::StatusOr<Responses> SessionBasic::RunDecode() {
  ABSL_LOG(INFO) << "RunDecodeSync";
  absl::StatusOr<Responses> responses;
  RETURN_IF_ERROR(worker_thread_pool_.Schedule(
      [this, &responses]() { responses = this->DecodeInternal(); }));
  RETURN_IF_ERROR(worker_thread_pool_.WaitUntilDone(Engine::kDefaultTimeout));
  return responses;
}

absl::Status SessionBasic::RunDecodeAsync(InferenceObservable* observer) {
  ABSL_LOG(INFO) << "RunDecodeAsync";
  return worker_thread_pool_.Schedule([this, observer]() {
    this->DecodeInternalStreaming(observer).IgnoreError();
  });
}

absl::StatusOr<Responses> SessionBasic::GenerateContent(
    const std::vector<InputData>& contents) {
  RETURN_IF_ERROR(RunPrefill(contents));
  return RunDecode();
}

absl::Status SessionBasic::GenerateContentStream(
    const std::vector<InputData>& contents, InferenceObservable* observer) {
  // An observer to handle the result of the async prefill operation.
  // It triggers the decode step if prefill is successful, or propagates the
  // error.
  class PrefillObserver : public InferenceObservable {
   public:
    PrefillObserver(SessionBasic* session, InferenceObservable* decode_observer)
        : session_(session), decode_observer_(decode_observer) {}

    void OnNext(const Responses& responses) override {
      ABSL_LOG(WARNING) << "OnNext should not be called during prefill!";
    }

    void OnError(const absl::Status& status) override {
      decode_observer_->OnError(status);
      delete this;
    }

    void OnDone() override {
      absl::Status status = session_->RunDecodeAsync(decode_observer_);
      if (!status.ok()) {
        decode_observer_->OnError(status);
      }
      delete this;
    }

   private:
    SessionBasic* session_;
    InferenceObservable* decode_observer_;
  };

  auto* prefill_observer = new PrefillObserver(this, observer);
  auto status = RunPrefillAsync(contents, prefill_observer);
  if (!status.ok()) {
    delete prefill_observer;
  }
  return status;
}

absl::StatusOr<BenchmarkInfo> SessionBasic::GetBenchmarkInfo() {
  if (benchmark_info_.has_value()) {
    return benchmark_info_.value();
  }
  return absl::InternalError(
      "Benchmark is not enabled. Please make sure the BenchmarkParams is set "
      "in the EngineSettings.");
}

}  // namespace litert::lm
