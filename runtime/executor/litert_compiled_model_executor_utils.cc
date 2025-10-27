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

#include "runtime/executor/litert_compiled_model_executor_utils.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <iterator>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_element_type.h"  // from @litert
#include "litert/cc/litert_expected.h"  // from @litert
#include "litert/cc/litert_model.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/components/model_resources.h"
#include "runtime/components/model_resources_litert_lm.h"
#include "runtime/components/model_resources_task.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/util/file_format_util.h"
#include "runtime/util/litert_lm_loader.h"
#include "runtime/util/model_asset_bundle_resources.h"
#include "runtime/util/scoped_file.h"
#include "runtime/util/status_macros.h"  //NOLINT

namespace litert::lm {

namespace {

// The name of the prefill decode model in the task bundle.
constexpr char kPrefilDecodeModelNameInTaskBundle[] = "TF_LITE_PREFILL_DECODE";
// Possible input tokens names:
constexpr std::array<absl::string_view, 2> kInputTokensNames = {"token_ids",
                                                                "tokens"};
// Possible input positions names:
constexpr std::array<absl::string_view, 2> kInputPositionsNames = {"positions",
                                                                   "input_pos"};
// Possible input attention mask names:
constexpr std::array<absl::string_view, 2> kInputAttnMaskNames = {"attn_mask",
                                                                  "mask"};
// Possible embedding names:
constexpr std::array<absl::string_view, 1> kEmbeddingNames = {"embeddings"};
// Possible per layer embedding names:
constexpr std::array<absl::string_view, 1> kPerLayerEmbeddingNames = {
    "per_layer_embeddings"};
// Possible output logits names:
constexpr std::array<absl::string_view, 1> kOutputLogitsNames = {"logits"};

absl::StatusOr<std::unique_ptr<ModelResources>>
BuildModelResourcesFromTaskFormat(std::shared_ptr<ScopedFile> model_file) {
  ASSIGN_OR_RETURN(auto resources,  // NOLINT
                   ModelAssetBundleResources::Create(/*tag=*/"", model_file));

  auto files_list = resources->ListFiles();
  RET_CHECK(std::find(files_list.begin(), files_list.end(),  // NOLINT
                      kPrefilDecodeModelNameInTaskBundle) != files_list.end())
      << kPrefilDecodeModelNameInTaskBundle
      << " model file not found in task bundle.";
  return ModelResourcesTask::Create(std::move(resources));
}

absl::StatusOr<std::unique_ptr<ModelResources>>
BuildModelResourcesFromLitertLmFormat(ScopedFile model_file) {
  auto loader = std::make_unique<LitertLmLoader>(std::move(model_file));

  ABSL_LOG(INFO) << "Read litert model from section.";

  // Save the loader for future use and keep the model alive.
  return ModelResourcesLitertLm::Create(std::move(loader));
}

}  // namespace

absl::StatusOr<ModelSignatures> GetModelSignaturesFromInputOutputNames(
    const std::vector<absl::string_view>& input_names,
    const std::vector<absl::string_view>& output_names, bool strict) {
  ModelSignatures model_signatures;
  for (auto input_name : input_names) {
    if (absl::c_linear_search(kInputTokensNames, input_name)) {
      model_signatures.input_tokens = std::string(input_name);
      continue;
    }
    if (absl::c_linear_search(kInputPositionsNames, input_name)) {
      model_signatures.input_positions = std::string(input_name);
      continue;
    }
    if (absl::c_linear_search(kInputAttnMaskNames, input_name)) {
      model_signatures.input_attn_mask = std::string(input_name);
      continue;
    }
    if (absl::c_linear_search(kEmbeddingNames, input_name)) {
      model_signatures.input_embeddings = std::string(input_name);
      continue;
    }
    if (absl::c_linear_search(kPerLayerEmbeddingNames, input_name)) {
      model_signatures.input_per_layer_embeddings = std::string(input_name);
      continue;
    }
  }

  for (auto output_name : output_names) {
    if (absl::c_linear_search(kOutputLogitsNames, output_name)) {
      model_signatures.output_logits = std::string(output_name);
      continue;
    }
  }

  if (strict) {
    RET_CHECK(!model_signatures.input_tokens.empty() ||
              model_signatures.input_embeddings.has_value())
            .SetCode(absl::StatusCode::kFailedPrecondition)
        << "Input tokens or embeddings not found.";
    RET_CHECK(!model_signatures.input_positions.empty())
            .SetCode(absl::StatusCode::kFailedPrecondition)
        << "Input positions not found.";
    RET_CHECK(!model_signatures.output_logits.empty())
            .SetCode(absl::StatusCode::kFailedPrecondition)
        << "Output logits not found.";
  }
  return model_signatures;
}

absl::StatusOr<SortedPrefillSignatureMap> GetPrefillRunnerSetFromModel(
    const ::litert::Model& model, const std::string& signature_name_base,
    const std::string& input_positions_name) {
  SortedPrefillSignatureMap prefill_runner_set;
  auto signatures = model.GetSignatures();
  for (auto& signature : *signatures) {
    if (auto signature_key = signature.Key();
        absl::StartsWith(signature_key, signature_name_base)) {
      auto input_positions_tensor = signature.InputTensor(input_positions_name);
      if (!input_positions_tensor) {
        return absl::InternalError(input_positions_tensor.Error().Message());
      }
      auto ranked_tensor_type = input_positions_tensor->RankedTensorType();
      if (!ranked_tensor_type) {
        return absl::InternalError(ranked_tensor_type.Error().Message());
      }

      if (ranked_tensor_type->Layout().Rank() == 2) {
        // [batch_size, max_seq_len]
        prefill_runner_set[ranked_tensor_type->Layout().Dimensions()[1]] =
            std::string(signature_key);
      } else if (ranked_tensor_type->Layout().Rank() == 1) {
        // [max_seq_len]
        prefill_runner_set[ranked_tensor_type->Layout().Dimensions()[0]] =
            std::string(signature_key);
      } else {
        return absl::FailedPreconditionError(
            "Unsupported input tokens tensor dimension.");
      }
    }
  }
  return prefill_runner_set;
}

absl::StatusOr<std::vector<std::pair<std::string, int>>>
GetOptimizedPrefillWorkGroups(
    const SortedPrefillSignatureMap& prefill_runner_set, int input_length) {
  std::vector<std::pair<std::string, int>> work_groups;
  // Current strategy:
  // 1. Use the prefill runner with the largest sequence length, until the
  // remaining length is less than its sequence length.
  // 2. Finish the remaining length with one prefill call, using the runner with
  // the sequence length as small as possible.
  // TODO: b/378772479 - Improve this strategy once we have benchmarked costs.
  int max_seq_len = prefill_runner_set.begin()->first;
  while (input_length >= max_seq_len) {
    work_groups.push_back(
        std::make_pair(prefill_runner_set.begin()->second, max_seq_len));
    input_length -= max_seq_len;
  }
  if (input_length > 0) {
    for (auto it = prefill_runner_set.begin(); it != prefill_runner_set.end();
         ++it) {
      // If the next smaller runner can handle the remaining length, skip the
      // current runner.
      if (std::next(it) != prefill_runner_set.end() &&
          std::next(it)->first >= input_length) {
        continue;
      }
      work_groups.push_back(std::make_pair(it->second, input_length));
      break;
    }
  }
  return work_groups;
}

absl::Status InitializeAttentionMask(litert::TensorBuffer& mask, bool is_f16) {
  auto mask_size = mask.PackedSize();
  RET_CHECK(mask_size) << "Failed to get attention mask buffer size.";
  auto mask_tensor_type = mask.TensorType();
  RET_CHECK(mask_tensor_type) << "Failed to get attention mask tensor type.";
  auto mask_lock_and_addr = litert::TensorBufferScopedLock::Create(
      mask, litert::TensorBuffer::LockMode::kWrite);
  RET_CHECK(mask_lock_and_addr) << "Failed to lock attention mask buffer.";

  switch (mask_tensor_type->ElementType()) {
    case litert::ElementType::Bool: {
      // Boolean mask: Default value = false.
      memset(mask_lock_and_addr->second, 0, *mask_size);
    } break;
    case litert::ElementType::Float32: {
      // Float mask: Default value is based on precision.
      // Default value reference:
      // third_party/odml/infra/genai/inference/ml_drift/llm/tasks/apply_attention_mask_test_util.cc
      float* mask_ptr = static_cast<float*>(mask_lock_and_addr->second);
      std::fill(mask_ptr, mask_ptr + *mask_size / sizeof(float),
                is_f16 ? -45824 : -0.7f * std::numeric_limits<float>::max());
    } break;
    default:
      return absl::InvalidArgumentError(
          "Unsupported attention mask data type.");
  }
  return absl::OkStatus();
}

absl::Status FillAttentionMask(litert::TensorBuffer& mask, int start_timestep,
                               int steps) {
  auto mask_tensor_type = mask.TensorType();
  RET_CHECK(mask_tensor_type) << "Failed to get attention mask tensor type.";
  RET_CHECK_EQ(mask_tensor_type->Layout().Rank(), 4)
          .SetCode(absl::StatusCode::kInvalidArgument)
      << "Attention mask must be 4D.";
  int channel_size = mask_tensor_type->Layout().Dimensions()[3];
  auto mask_lock_and_addr = litert::TensorBufferScopedLock::Create(
      mask, litert::TensorBuffer::LockMode::kWrite);
  RET_CHECK(mask_lock_and_addr) << "Failed to lock attention mask buffer.";

  for (int i = 0; i < steps; ++i) {
    int current_step = start_timestep + i;
    int offset = i * channel_size;
    // For current step = n, we fill (n+1) positions for the mask sequence.
    switch (mask_tensor_type->ElementType()) {
      case litert::ElementType::Bool: {
        // Boolean mask: Fill value = true.
        bool* mask_bool_ptr = static_cast<bool*>(mask_lock_and_addr->second);
        std::fill(mask_bool_ptr + offset,
                  mask_bool_ptr + offset + current_step + 1, true);
      } break;
      case litert::ElementType::Float32: {
        // Float mask: Fill value = 0.0f.
        float* mask_float_ptr = static_cast<float*>(mask_lock_and_addr->second);
        std::fill(mask_float_ptr + offset,
                  mask_float_ptr + offset + current_step + 1, 0.0f);
      } break;
      default:
        return absl::InvalidArgumentError(
            "Unsupported attention mask data type.");
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<ModelResources>>
BuildLiteRtCompiledModelResources(const ModelAssets& model_assets) {
  ASSIGN_OR_RETURN(  // NOLINT
      auto format,
      GetFileFormat(model_assets.GetPath().value_or(""),
                    model_assets.GetScopedFile().value_or(nullptr)));

  ASSIGN_OR_RETURN(auto scoped_file,  // NOLINT
                   model_assets.GetOrCreateScopedFile());

  switch (format) {
    case FileFormat::TFLITE:
      return absl::InvalidArgumentError("Unsupported file format.");
    case FileFormat::TASK:
      return BuildModelResourcesFromTaskFormat(std::move(scoped_file));
    case FileFormat::LITERT_LM: {
      // `BuildModelResourcesFromLitertLmFormat` expects a ScopedFile that it
      // takes ownership of, so we need to duplicate the ScopedFile to keep
      // the original alive.
      ASSIGN_OR_RETURN(auto duplicate_file, scoped_file->Duplicate());
      return BuildModelResourcesFromLitertLmFormat(std::move(duplicate_file));
    }
  }
}

}  // namespace litert::lm
