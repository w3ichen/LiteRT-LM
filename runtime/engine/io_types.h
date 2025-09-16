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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_ENGINE_IO_TYPES_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_ENGINE_IO_TYPES_H_

#include <cstdint>
#include <map>
#include <ostream>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/proto/engine.pb.h"

namespace litert::lm {

// A container to host the input text.
class InputText {
 public:
  // Constructs an InputText from a raw text string or a TensorBuffer of token
  // ids. The InputText takes ownership of the provided data.
  explicit InputText(std::variant<std::string, TensorBuffer> data)
      : data_(std::move(data)) {}

  // Copy constructor.
  InputText(const InputText& other) = delete;
  // Copy assignment operator.
  InputText& operator=(const InputText& other) = delete;
  // Move constructor.
  InputText(InputText&& other) = default;
  // Move assignment operator.
  InputText& operator=(InputText&& other) = default;

  // Returns true if the text is preprocessed into a TensorBuffer.
  bool IsTensorBuffer() const {
    return std::holds_alternative<TensorBuffer>(data_);
  }

  // Returns the raw text string. Returns an error if the text is preprocessed.
  absl::StatusOr<absl::string_view> GetRawTextString() const {
    if (std::holds_alternative<std::string>(data_)) {
      return absl::string_view(std::get<std::string>(data_));
    }
    return absl::FailedPreconditionError(
        "The text is preprocessed and does not have raw text bytes.");
  }

  // Returns the preprocessed text tensor. Returns an error if the text is
  // not preprocessed.
  absl::StatusOr<const TensorBuffer*> GetPreprocessedTextTensor() const {
    if (std::holds_alternative<TensorBuffer>(data_)) {
      return &std::get<TensorBuffer>(data_);
    }
    return absl::FailedPreconditionError(
        "The text is not preprocessed and does not have a tensor.");
  }

 private:
  std::variant<std::string, TensorBuffer> data_;
};

// A container to host the input image.
class InputImage {
 public:
  // Constructs an InputImage from a raw image bytes string or a TensorBuffer of
  // processed image bytes. The InputImage takes ownership of the provided data.
  explicit InputImage(std::variant<std::string, TensorBuffer> data)
      : data_(std::move(data)) {}

  // Copy constructor.
  InputImage(const InputImage& other) = delete;
  // Copy assignment operator.
  InputImage& operator=(const InputImage& other) = delete;
  // Move constructor.
  InputImage(InputImage&& other) = default;
  // Move assignment operator.
  InputImage& operator=(InputImage&& other) = default;

  // Returns true if the image is preprocessed into a TensorBuffer.
  bool IsTensorBuffer() const {
    return std::holds_alternative<TensorBuffer>(data_);
  }

  // Returns the raw image bytes. Returns an error if the image is preprocessed.
  absl::StatusOr<absl::string_view> GetRawImageBytes() const {
    if (std::holds_alternative<std::string>(data_)) {
      return absl::string_view(std::get<std::string>(data_));
    }
    return absl::FailedPreconditionError(
        "The image is preprocessed and does not have raw image bytes.");
  }

  // Returns the preprocessed image tensor. Returns an error if the image is
  // not preprocessed.
  absl::StatusOr<const TensorBuffer*> GetPreprocessedImageTensor() const {
    if (std::holds_alternative<TensorBuffer>(data_)) {
      return &std::get<TensorBuffer>(data_);
    }
    return absl::FailedPreconditionError(
        "The image is not preprocessed and does not have a tensor.");
  }

 private:
  std::variant<std::string, TensorBuffer> data_;
};

// A container to host the input audio.
class InputAudio {
 public:
  // Constructs an InputAudio from a raw audio bytes string or a TensorBuffer of
  // processed audio bytes. The InputAudio takes ownership of the provided data.
  explicit InputAudio(std::variant<std::string, TensorBuffer> data)
      : data_(std::move(data)) {}

  // Copy constructor.
  InputAudio(const InputAudio& other) = delete;
  // Copy assignment operator.
  InputAudio& operator=(const InputAudio& other) = delete;
  // Move constructor.
  InputAudio(InputAudio&& other) = default;
  // Move assignment operator.
  InputAudio& operator=(InputAudio&& other) = default;

  // Returns true if the audio is preprocessed into a TensorBuffer.
  bool IsTensorBuffer() const {
    return std::holds_alternative<TensorBuffer>(data_);
  }

  // Returns the raw audio bytes. Returns an error if the audio is preprocessed.
  absl::StatusOr<absl::string_view> GetRawAudioBytes() const {
    if (std::holds_alternative<std::string>(data_)) {
      return absl::string_view(std::get<std::string>(data_));
    }
    return absl::FailedPreconditionError(
        "The audio is preprocessed and does not have raw audio bytes.");
  }

  // Returns the preprocessed audio tensor. Returns an error if the audio is
  // not preprocessed.
  absl::StatusOr<const TensorBuffer*> GetPreprocessedAudioTensor() const {
    if (std::holds_alternative<TensorBuffer>(data_)) {
      return &std::get<TensorBuffer>(data_);
    }
    return absl::FailedPreconditionError(
        "The audio is not preprocessed and does not have a tensor.");
  }

 private:
  std::variant<std::string, TensorBuffer> data_;
};

// A container to host the input data. Will be extended to support more input
// types in the future.
using InputData = std::variant<InputText, InputImage, InputAudio>;

// A container to host the model responses.
class Responses {
 public:
  explicit Responses(int num_output_candidates);

  // Returns the number of output candidates.
  int GetNumOutputCandidates() const { return num_output_candidates_; }

  // Returns the response text at the given index. Returns error if the index is
  // out of range.
  absl::StatusOr<absl::string_view> GetResponseTextAt(int index) const;

  // Returns the score at the given index. Returns error if the index is out of
  // range or if scores are not included.
  // Note that the "score" is calculated as the sum of the log probabilities of
  // the whole decoded sequence normalized by the total number of tokens.
  absl::StatusOr<float> GetScoreAt(int index) const;

  // Returns the mutable response texts vector.
  std::vector<std::string>& GetMutableResponseTexts();

  // Returns the mutable scores vector. If it is the first time calling this
  // function, the scores vector will be allocated to the size of
  // num_output_candidates_ and initialized to the default value of -Inf
  // (= log(0.0f)).
  std::vector<float>& GetMutableScores();

 private:
  // The number of output candidates.
  int num_output_candidates_;

  // The output vector of response tokens (as strings).
  std::vector<std::string> response_texts_;

  // The output vector of scores for each response text. The "score" is pulled
  // from the probability of the last token in the response text.
  std::vector<float> scores_;
};
std::ostream& operator<<(std::ostream& os, const Responses& responses);

// Class to store the data for a single turn of the benchmark. A "turn" is
// defined as a single RunPrefill or RunDecode call.
struct BenchmarkTurnData {
  absl::Duration duration;  // Duration of this entire operation/turn.
  uint64_t num_tokens;      // The number of tokens processed in this turn.
  BenchmarkTurnData(uint64_t tokens, absl::Duration dur);
};
std::ostream& operator<<(std::ostream& os, const BenchmarkTurnData& data);

// Class to store and manage comprehensive performance benchmark information for
// LLMs.
class BenchmarkInfo {
 public:
  explicit BenchmarkInfo(const proto::BenchmarkParams& benchmark_params);
  const proto::BenchmarkParams& GetBenchmarkParams() const;

  // --- Methods to record data ---
  // Time the start and end of a phase in the initialization. The phase name
  // should be a string that uniquely identifies the phase. Otherwise, the
  // methods will return an error.
  absl::Status TimeInitPhaseStart(const std::string& phase_name);
  absl::Status TimeInitPhaseEnd(const std::string& phase_name);
  // Time the start and end of a prefill/decode turn. The num_prefill_tokens
  // should be the number of tokens processed in this turn. The method will
  // return an error if the methods are called out of order (i.e. one end after
  // one start).
  absl::Status TimePrefillTurnStart();
  absl::Status TimePrefillTurnEnd(uint64_t num_prefill_tokens);
  absl::Status TimeDecodeTurnStart();
  absl::Status TimeDecodeTurnEnd(uint64_t num_decode_tokens);
  // Time the duration between two consecutive marks. Useful for profiling the
  // pipeline at a specific point. For example:
  //   RETURN_IF_ERROR(benchmark_info.TimeMarkDelta("sampling"));
  //   ... actual sampling logics ...
  //   RETURN_IF_ERROR(benchmark_info.TimeMarkDelta("sampling"));
  //
  // The method will return the duration as the time delta between the two
  // TimeMarkDelta("sampling") calls. The duration will be stored / recorded for
  // each unique mark name.
  absl::Status TimeMarkDelta(const std::string& mark_name);

  // --- Getters for raw data ---
  const std::map<std::string, absl::Duration>& GetInitPhases() const;
  const std::map<std::string, absl::Duration>& GetMarkDurations() const;

  // --- Calculated metrics and getters for Prefill ---
  uint64_t GetTotalPrefillTurns() const;
  const BenchmarkTurnData& GetPrefillTurn(int turn_index) const;
  double GetPrefillTokensPerSec(int turn_index) const;

  // --- Calculated metrics and getters for Decode ---
  uint64_t GetTotalDecodeTurns() const;
  const BenchmarkTurnData& GetDecodeTurn(int turn_index) const;
  double GetDecodeTokensPerSec(int turn_index) const;

  // --- Gets the time to the first token ---
  // Note that the first time to token doesn't include the time for
  // initialization. It is the sum of the prefill time for the first turn and
  // the time spent for decoding the first token.
  double GetTimeToFirstToken() const;

 private:
  proto::BenchmarkParams benchmark_params_;

  // Map of phase names to start time.
  std::map<std::string, absl::Time> start_time_map_;
  std::map<std::string, absl::Time> mark_time_map_;
  // The current index of the prefill / decode turn.
  int prefill_turn_index_ = 0;
  int decode_turn_index_ = 0;

  std::map<std::string, absl::Duration> init_phases_;
  std::map<std::string, absl::Duration> mark_durations_;
  std::vector<BenchmarkTurnData> prefill_turns_;
  std::vector<BenchmarkTurnData> decode_turns_;
};
std::ostream& operator<<(std::ostream& os, const BenchmarkInfo& info);

// An interface with default implementations for the inference observer. The
// default implementations are print out the first response text and the final
// (or error) status to the stderr.
class InferenceObservable {
 public:
  virtual ~InferenceObservable() = default;

  // Called when a new response is generated.
  virtual void OnNext(const Responses& responses);

  // Called when the inference is done and finished successfully.
  virtual void OnDone();

  // Called when an error is encountered during the inference.
  virtual void OnError(const absl::Status& status);
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_ENGINE_IO_TYPES_H_
