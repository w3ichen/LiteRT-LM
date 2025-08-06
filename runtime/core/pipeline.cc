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

#include "runtime/core/pipeline.h"

#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_replace.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/components/sampler.h"
#include "runtime/components/stop_token_detector.h"
#include "runtime/components/tokenizer.h"
#include "runtime/engine/io_types.h"
#include "runtime/executor/llm_executor.h"
#include "runtime/executor/llm_executor_io_types.h"
#include "runtime/util/convert_tensor_buffer.h"
#include "runtime/util/litert_status_util.h"
#include "runtime/util/status_macros.h"  //NOLINT

namespace litert::lm {
namespace {

// TODO(b/423364170): all LLM Executors should respect the max number of tokens
// returned by the model. We should remove this default value once all Executors
// are compliant with the max number of tokens.
constexpr int kDefaultMaxNumTokens = 4096;
int TryGetMaxNumTokens(const LlmExecutor& executor) {
  auto settings = executor.GetExecutorSettings();
  if (!settings.ok()) {
    // If the executor settings are not available, we will use the default
    // value.
    ABSL_LOG(WARNING) << "Failed to get executor settings: "
                      << settings.status();
    return kDefaultMaxNumTokens;
  }
  return settings->GetMaxNumTokens();
}

// Check whether the decoding loop should stop.
bool ShouldStop(bool hit_stop_tokens, int benchmark_decode_token_count,
                int num_decoded_steps, int current_step, int max_num_tokens,
                InferenceObservable* observer) {
  // Stopping conditions.
  if (hit_stop_tokens && benchmark_decode_token_count == 0) {
    // Only early stop if no decode step
    // is requested by benchmark.
    return true;
  } else if (benchmark_decode_token_count > 0 &&
             num_decoded_steps >= benchmark_decode_token_count) {
    // Stop when the number of decode steps is equal to the
    // benchmark_decode_token_count (when specified).
    return true;
  } else if (current_step >= max_num_tokens) {
    // Reaching maximum number of kv-cache size.
    if (observer != nullptr) {
      observer->OnError(absl::InternalError("Maximum kv-cache size reached."));
    }
    return true;
  }
  return false;
}

// The result of a invocation of the decode process for a single batch of
// tokens.
// kPartial indicates that at least one output candidate needs to be re-decoded
// with additional tokens, while kDone indicates that all output candidates are
// complete. kContinue represents the steady state of the decoding loop.
enum DecodeResult {
  kPartial,   // BPE token encountered, need more tokens to complete decoding.
  kContinue,  // Next token decoded, but no stop token encountered.
  kDone,      // Stop token encountered, decoding is complete.
};

// A wrapper class to run one step of the decode process, handling both internal
// and external sampling.
class DecodeOneStep {
 public:
  DecodeOneStep(LlmExecutor* absl_nonnull executor,
                Tokenizer* absl_nonnull tokenizer, int num_output_candidates,
                const StopTokenDetector& stop_token_detector,
                std::optional<BenchmarkInfo>& benchmark_info,
                std::optional<Sampler*> sampler)
      : executor_(*executor),
        tokenizer_(*tokenizer),
        num_output_candidates_(num_output_candidates),
        sampler_(sampler),
        benchmark_info_(benchmark_info),
        stop_token_detector_(stop_token_detector) {
    if (!sampler_.has_value()) {  // Internal sampling setup
      auto output_tokens = CreateTensorBuffer<int>({num_output_candidates_, 1});
      output_tokens_ = std::move(*output_tokens);
    } else {  // External sampling setup
      auto scores_tensor = CreateTensorBuffer<float>({num_output_candidates_});
      scores_tensor_ = std::move(*scores_tensor);
    }
  }

  // Runs one step of the decode process.
  // For external sampling, `decoded_ids` must be provided and will be updated.
  // For internal sampling, `decoded_ids` is ignored.
  absl::StatusOr<DecodeResult> Run(
      std::optional<litert::TensorBuffer*> decoded_ids = std::nullopt) {
    ASSIGN_OR_RETURN(litert::TensorBuffer * next_tokens_buffer,
                     DecodeAndSample(decoded_ids));

    // Post-processing the next tokens.
    ASSIGN_OR_RETURN(auto token_ids,
                     tokenizer_.TensorBufferToTokenIds(*next_tokens_buffer));
    ASSIGN_OR_RETURN(token_ids_, previous_token_ids_.empty()
                                     ? token_ids
                                     : tokenizer_.MergeTokenIds(
                                           previous_token_ids_, token_ids));

    auto decoded_result =
        tokenizer_.TokenIdsToTexts(num_output_candidates_, token_ids_);

    if (Tokenizer::IsIncompleteBpeSequence(decoded_result)) {
      previous_token_ids_ = token_ids_;
      return kPartial;
    }
    // Empty the previous token IDs buffer for the next step.
    previous_token_ids_.clear();
    ASSIGN_OR_RETURN(result_text_, decoded_result);

    LITERT_ASSIGN_OR_RETURN_ABSL(
        auto next_tokens_span,
        ReferTensorBufferAsSpan<int>(*next_tokens_buffer));
    RETURN_IF_ERROR(stop_token_detector_.ProcessTokens(next_tokens_span));

    if (sampler_.has_value()) {
      LITERT_ASSIGN_OR_RETURN_ABSL(
          scores_span_, ReferTensorBufferAsSpan<float>(scores_tensor_));
    }

    ASSIGN_OR_RETURN(bool hit_stop_tokens, stop_token_detector_.AllDone());
    return hit_stop_tokens ? kDone : kContinue;
  }

  absl::Span<float> GetScores() { return scores_span_; }

  const std::vector<std::string>& GetResultText() const { return result_text_; }
  const std::vector<bool>& GetStopTokensFound() const {
    return stop_token_detector_.GetStopTokensFound();
  }

  bool IsPartialStopTokenFound(int index) const {
    return stop_token_detector_.IsPartialStopTokenFound(index);
  }
  const std::vector<std::vector<int>>& GetTokenIds() const {
    return token_ids_;
  }

 private:
  // Runs the core decoding and sampling step, for either internal or external
  // sampling. Returns a pointer to the tensor buffer containing the next token
  // IDs.
  absl::StatusOr<litert::TensorBuffer*> DecodeAndSample(
      std::optional<litert::TensorBuffer*> decoded_ids) {
    if (sampler_) {  // External sampling path
      if (!decoded_ids) {
        return absl::InternalError(
            "decoded_ids must be provided for external sampling.");
      }
      LITERT_ASSIGN_OR_RETURN(auto duplicate_decoded_ids,
                              decoded_ids.value()->Duplicate());
      ExecutorInputs inputs(ExecutorTextData(std::move(duplicate_decoded_ids)),
                            std::nullopt, std::nullopt);
      // Decoding section.
      if (benchmark_info_.has_value()) {
        RETURN_IF_ERROR(benchmark_info_->TimeMarkDelta("executor_decode"));
      }
      ASSIGN_OR_RETURN(auto output_logits, executor_.DecodeLogits(inputs));
      if (benchmark_info_.has_value()) {
        RETURN_IF_ERROR(benchmark_info_->TimeMarkDelta("executor_decode"));
      }

      // Samping section.
      if (benchmark_info_.has_value()) {
        RETURN_IF_ERROR(benchmark_info_->TimeMarkDelta("sampling"));
      }
      RETURN_IF_ERROR(sampler_.value()->SampleToIdAndScoreBuffer(
          output_logits, *decoded_ids.value(), &scores_tensor_));
      if (benchmark_info_.has_value()) {
        RETURN_IF_ERROR(benchmark_info_->TimeMarkDelta("sampling"));
      }

      return decoded_ids.value();
    } else {  // Internal sampling path
      // Benchmark executor_decode_and_sample section.
      if (benchmark_info_.has_value()) {
        RETURN_IF_ERROR(
            benchmark_info_->TimeMarkDelta("executor_decode_and_sample"));
      }
      RETURN_IF_ERROR(executor_.Decode(output_tokens_));
      if (benchmark_info_.has_value()) {
        RETURN_IF_ERROR(
            benchmark_info_->TimeMarkDelta("executor_decode_and_sample"));
      }
      return &output_tokens_;
    }
  }

  LlmExecutor& executor_;
  Tokenizer& tokenizer_;
  const int num_output_candidates_;
  std::optional<Sampler*> sampler_;
  std::optional<BenchmarkInfo> benchmark_info_;
  StopTokenDetector stop_token_detector_;

  // For internal sampling.
  // Holds the output token IDs. Dim: {num_output_candidates, 1}
  litert::TensorBuffer output_tokens_;

  // For external sampling.
  // Holds the scores for the output candidates. Dim: {num_output_candidates}
  litert::TensorBuffer scores_tensor_;
  absl::Span<float> scores_span_;

  // Common state
  std::vector<std::vector<int>> previous_token_ids_;
  std::vector<std::vector<int>> token_ids_;
  std::vector<std::string> result_text_;
};

absl::StatusOr<Responses> DecodeLoop(
    LlmExecutor& executor, Tokenizer& tokenizer,
    const StopTokenDetector& stop_token_detector, int num_output_candidates,
    std::optional<BenchmarkInfo>& benchmark_info,
    std::optional<Sampler*> sampler,
    std::optional<litert::TensorBuffer*> decoded_ids,
    std::optional<InferenceObservable*> observer) {
  const bool is_streaming = observer.has_value();
  const bool is_custom_sampling = sampler.has_value();

  int benchmark_decode_token_count = 0;
  if (benchmark_info.has_value()) {
    benchmark_decode_token_count =
        benchmark_info->GetBenchmarkParams().num_decode_tokens();
    RETURN_IF_ERROR(benchmark_info->TimeDecodeTurnStart());
  }

  Responses final_responses(num_output_candidates);
  std::vector<float> accumulated_scores(num_output_candidates, 0.0f);
  std::vector<int> num_decoded_tokens(num_output_candidates, 0);

  int num_decode_steps = 0;
  const int max_num_tokens = TryGetMaxNumTokens(executor);
  DecodeOneStep run_one_step(&executor, &tokenizer, num_output_candidates,
                             stop_token_detector, benchmark_info, sampler);

  std::vector<std::string> pending_stop_tokens(num_output_candidates);
  while (true) {
    absl::StatusOr<DecodeResult> decode_result = run_one_step.Run(decoded_ids);
    if (!decode_result.ok()) {
      if (is_streaming) observer.value()->OnError(decode_result.status());
      return decode_result.status();
    }

    if (*decode_result == kPartial) {
      continue;
    }
    num_decode_steps++;

    Responses step_responses(num_output_candidates);
    bool any_updates = false;
    for (int j = 0; j < num_output_candidates; ++j) {
      if (run_one_step.GetStopTokensFound()[j]) {
        continue;
      }

      if (run_one_step.IsPartialStopTokenFound(j)) {
        pending_stop_tokens[j] += run_one_step.GetResultText()[j];
        continue;
      }

      any_updates = true;
      // The tokenizer may return a token with a special character " " that
      // should be replaced with a space.
      std::string result_text = absl::StrReplaceAll(
          (pending_stop_tokens[j] + run_one_step.GetResultText()[j]),
          {{"▁", " "}});
      if (is_streaming) {
        step_responses.GetMutableResponseTexts()[j] = result_text;
        if (is_custom_sampling) {
          step_responses.GetMutableScores()[j] = run_one_step.GetScores()[j];
        }
      } else {
        final_responses.GetMutableResponseTexts()[j] += result_text;
        if (is_custom_sampling) {
          accumulated_scores[j] += run_one_step.GetScores()[j];
          num_decoded_tokens[j]++;
        }
      }
      // Clear the pending stop tokens for the next step.
      pending_stop_tokens[j].clear();
    }

    if (is_streaming && any_updates && *decode_result == kContinue) {
      observer.value()->OnNext(step_responses);
    }

    if (ShouldStop(*decode_result == kDone, benchmark_decode_token_count,
                   num_decode_steps, executor.GetCurrentStep().value(),
                   max_num_tokens, observer.value_or(nullptr))) {
      break;
    }
  }

  if (benchmark_info.has_value()) {
    RETURN_IF_ERROR(benchmark_info->TimeDecodeTurnEnd(num_decode_steps *
                                                      num_output_candidates));
  }

  if (is_streaming) {
    observer.value()->OnDone();
    return Responses(0);  // Return empty response for streaming.
  }

  // Finalize scores for non-streaming custom sampling.
  if (is_custom_sampling) {
    std::vector<float>& final_scores = final_responses.GetMutableScores();
    for (int j = 0; j < num_output_candidates; ++j) {
      if (num_decoded_tokens[j] > 0) {
        final_scores[j] = accumulated_scores[j] / num_decoded_tokens[j];
      } else {
        final_scores[j] = -std::numeric_limits<float>::infinity();
      }
    }
  }
  return final_responses;
}

}  // namespace

absl::StatusOr<int> Prefill(LlmExecutor& executor, Tokenizer& tokenizer,
                            absl::string_view prompt, int bos_token_id,
                            bool wait_for_completion,
                            std::optional<BenchmarkInfo>& benchmark_info) {
  int benchmark_prefill_token_count = 0;
  if (benchmark_info.has_value()) {
    benchmark_prefill_token_count =
        benchmark_info->GetBenchmarkParams().num_prefill_tokens();
    RETURN_IF_ERROR(benchmark_info->TimePrefillTurnStart());
  }
  ASSIGN_OR_RETURN(std::vector<int> ids, tokenizer.TextToTokenIds(prompt));
  if (benchmark_prefill_token_count > 0) {
    // If benchmark is enabled, we will use the benchmark prefill token count
    // to set the prefill token count.
    ids.resize(benchmark_prefill_token_count);
  } else {
    ids.insert(ids.begin(), bos_token_id);
  }
  const int max_num_tokens = TryGetMaxNumTokens(executor);
  if (ids.size() >= max_num_tokens) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Input token ids are too long. Exceeding the maximum number of tokens "
        "allowed: ",
        ids.size(), " >= ", max_num_tokens));
  }
  ASSIGN_OR_RETURN(auto ids_buffer, tokenizer.TokenIdsToTensorBuffer(ids));
  LITERT_ASSIGN_OR_RETURN_ABSL(auto ids_buffer_span,
                               ReferTensorBufferAsSpan<int>(ids_buffer));
  if (ids_buffer_span.empty()) {
    return absl::InternalError("Input token ids are empty.");
  }
  const int last_token_id = ids_buffer_span.back();
  ExecutorPrefillParams params;
  params.SetWaitForCompletion(wait_for_completion);
  RETURN_IF_ERROR(
      executor.Prefill(ExecutorInputs(ExecutorTextData(std::move(ids_buffer)),
                                      std::nullopt, std::nullopt),
                       params));
  if (benchmark_info.has_value()) {
    RETURN_IF_ERROR(benchmark_info->TimePrefillTurnEnd(ids_buffer_span.size()));
  }
  return last_token_id;
}

absl::StatusOr<Responses> Decode(LlmExecutor& executor, Tokenizer& tokenizer,
                                 const StopTokenDetector& stop_token_detector,
                                 std::optional<BenchmarkInfo>& benchmark_info) {
  const int num_output_candidates = 1;
  return DecodeLoop(executor, tokenizer, stop_token_detector,
                    num_output_candidates, benchmark_info, std::nullopt,
                    std::nullopt, std::nullopt);
}

absl::Status DecodeStreaming(LlmExecutor& executor, Tokenizer& tokenizer,
                             const StopTokenDetector& stop_token_detector,
                             std::optional<BenchmarkInfo>& benchmark_info,
                             InferenceObservable* observer) {
  if (observer == nullptr) {
    return absl::InvalidArgumentError(
        "Observer must not be null for streaming.");
  }
  const int num_output_candidates = 1;
  return DecodeLoop(executor, tokenizer, stop_token_detector,
                    num_output_candidates, benchmark_info, std::nullopt,
                    std::nullopt, observer)
      .status();
}

absl::StatusOr<Responses> DecodeCustomSampling(
    LlmExecutor& executor, Tokenizer& tokenizer,
    const StopTokenDetector& stop_token_detector, int num_output_candidates,
    Sampler& sampler, litert::TensorBuffer& decoded_ids,
    std::optional<BenchmarkInfo>& benchmark_info) {
  return DecodeLoop(executor, tokenizer, stop_token_detector,
                    num_output_candidates, benchmark_info, &sampler,
                    &decoded_ids, std::nullopt);
}

absl::Status DecodeCustomSamplingStreaming(
    LlmExecutor& executor, Tokenizer& tokenizer,
    const StopTokenDetector& stop_token_detector, int num_output_candidates,
    Sampler& sampler, litert::TensorBuffer& decoded_ids,
    std::optional<BenchmarkInfo>& benchmark_info,
    InferenceObservable* observer) {
  if (observer == nullptr) {
    return absl::InvalidArgumentError(
        "Observer must not be null for streaming.");
  }
  return DecodeLoop(executor, tokenizer, stop_token_detector,
                    num_output_candidates, benchmark_info, &sampler,
                    &decoded_ids, observer)
      .status();
}

}  // namespace litert::lm
