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

#include "runtime/executor/magic_number_configs_helper.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iterator>
#include <set>
#include <utility>
#include <vector>

#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/litert_environment_options.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_expected.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_model.h"  // from @litert
#include "runtime/executor/llm_executor_settings.h"

namespace litert::lm {
namespace {

struct MagicNumbers {
  int64_t context_length = 0;
  std::vector<int64_t> prefill_lengths;
};

constexpr absl::string_view kPrefillSignaturePrefix = "prefill";
constexpr absl::string_view kDecodeSignaturePrefix = "decode";
constexpr absl::string_view kTestPrefillSignaturePrefix = "test_prefill";
constexpr absl::string_view kTestDecodeSignaturePrefix = "test_decode";
constexpr absl::string_view kMaskSubstr = "mask";
constexpr absl::string_view kInputPosSubstr = "pos";
constexpr int64_t kDefaultTargetNumberBase = 256;

Expected<int64_t> GetLastDimension(const Subgraph& subgraph,
                                   absl::string_view input_name) {
  LITERT_ASSIGN_OR_RETURN(auto tensor, subgraph.Input(input_name));
  LITERT_ASSIGN_OR_RETURN(auto type, tensor.RankedTensorType());
  return type.Layout().Dimensions()[type.Layout().Rank() - 1];
}

// Check if the number is a magic number.
// The number is a magic number if it is prime and greater than 10.
bool IsMagicNumber(int64_t number) {
  if (number < 11) {
    return false;
  }
  if (number % 2 == 0) {
    return false;
  }
  for (int i = 3; i <= number / 2; i += 2) {
    if (number % i == 0) {
      return false;
    }
  }
  return true;
}

Expected<void> SetMagicNumberIfPrime(const Subgraph& subgraph,
                                     absl::string_view input_name,
                                     int64_t& magic_number) {
  LITERT_ASSIGN_OR_RETURN(auto dim, GetLastDimension(subgraph, input_name));
  if (IsMagicNumber(dim)) {
    if (magic_number == 0) {
      magic_number = dim;
    } else {
      LITERT_RETURN_IF_ERROR(magic_number == dim);
    }
  }
  return {};
}

// Returns the magic numbers from the model. prefill_lengths are sorted in
// ascending order.
Expected<MagicNumbers> GetMagicNumbersFromModel(const Model& litert_model) {
  auto num_signatures = litert_model.GetNumSignatures();
  MagicNumbers magic_numbers;
  for (int i = 0; i < num_signatures; ++i) {
    LITERT_ASSIGN_OR_RETURN(auto signature, litert_model.GetSignature(i));
    LITERT_ASSIGN_OR_RETURN(auto subgraph,
                            litert_model.Subgraph(signature.Key()));
    if (signature.Key().starts_with(kPrefillSignaturePrefix)) {
      for (const auto& input_name : signature.InputNames()) {
        if (absl::StrContains(input_name, kMaskSubstr)) {
          LITERT_RETURN_IF_ERROR(SetMagicNumberIfPrime(
              subgraph, input_name, magic_numbers.context_length));
        } else if (absl::StrContains(input_name, kInputPosSubstr)) {
          int64_t prefill_length = 0;
          LITERT_RETURN_IF_ERROR(
              SetMagicNumberIfPrime(subgraph, input_name, prefill_length));
          if (prefill_length > 0) {
            magic_numbers.prefill_lengths.push_back(prefill_length);
          }
        }
      }
    } else if (signature.Key().starts_with(kDecodeSignaturePrefix)) {
      for (const auto& input_name : signature.InputNames()) {
        if (absl::StrContains(input_name, kMaskSubstr)) {
          LITERT_RETURN_IF_ERROR(SetMagicNumberIfPrime(
              subgraph, input_name, magic_numbers.context_length));
        }
      }
    }
  }

  std::sort(magic_numbers.prefill_lengths.begin(),
            magic_numbers.prefill_lengths.end());
  return magic_numbers;
}

Expected<std::vector<std::pair<absl::string_view, absl::string_view>>>
GetVerificationPairs(const Model& litert_model,
                     const MagicNumbers& magic_numbers,
                     const MagicNumbers& target_numbers) {
  std::vector<std::pair<absl::string_view, absl::string_view>> verify_pairs;
  auto num_signatures = litert_model.GetNumSignatures();
  for (int i = 0; i < num_signatures; ++i) {
    LITERT_ASSIGN_OR_RETURN(auto signature, litert_model.GetSignature(i));
    if (!signature.Key().starts_with(kPrefillSignaturePrefix) &&
        !signature.Key().starts_with(kDecodeSignaturePrefix)) {
      continue;
    }

    LITERT_ASSIGN_OR_RETURN(auto subgraph,
                            litert_model.Subgraph(signature.Key()));
    for (int j = 0; j < num_signatures; ++j) {
      LITERT_ASSIGN_OR_RETURN(auto test_signature,
                              litert_model.GetSignature(j));
      if (!test_signature.Key().starts_with(kTestPrefillSignaturePrefix) &&
          !test_signature.Key().starts_with(kTestDecodeSignaturePrefix)) {
        continue;
      }

      LITERT_ASSIGN_OR_RETURN(auto test_subgraph,
                              litert_model.Subgraph(test_signature.Key()));
      bool is_same_shape = true;
      for (const auto& input_name : signature.InputNames()) {
        if (absl::StrContains(input_name, kMaskSubstr) ||
            absl::StrContains(input_name, kInputPosSubstr)) {
          LITERT_ASSIGN_OR_RETURN(auto dim,
                                  GetLastDimension(subgraph, input_name));
          LITERT_ASSIGN_OR_RETURN(auto test_dim,
                                  GetLastDimension(test_subgraph, input_name));
          // Check if dim is same as test_dim, or as a magic number when
          // test_dim is target number corresponding to the magic number.
          // Otherwise, the shapes are not same.
          if (dim != test_dim) {
            if (dim == magic_numbers.context_length &&
                test_dim == target_numbers.context_length) {
              continue;
            }
            auto it = std::find(magic_numbers.prefill_lengths.begin(),
                                magic_numbers.prefill_lengths.end(), dim);
            if (it != magic_numbers.prefill_lengths.end()) {
              size_t diff = it - magic_numbers.prefill_lengths.begin();
              if (test_dim == target_numbers.prefill_lengths[diff]) {
                continue;
              }
            }
            is_same_shape = false;
            break;
          }
        }
      }
      if (is_same_shape) {
        verify_pairs.push_back(
            std::make_pair(signature.Key(), test_signature.Key()));
      }
    }
  }
  return verify_pairs;
}

// Returns the default target number for the given magic number. The target
// number is the largest multiple of 256 smaller than the magic number.
// If the magic number is smaller than 256, it is the largest exponential number
// smaller than the magic number.
int64_t GetDefaultTargetNumber(int64_t magic_number) {
  if (magic_number > kDefaultTargetNumberBase) {
    return magic_number / kDefaultTargetNumberBase * kDefaultTargetNumberBase;
  }

  int64_t target_number = kDefaultTargetNumberBase;
  while (target_number > magic_number) {
    target_number /= 2;
  }
  return target_number;
}

// Returns the target number for the given magic number and target number hint.
// If the hint is less than magic number, it returns the hint. If the hint is
// larger than magic number, it returns the default target number.
int64_t GetTargetNumber(int64_t magic_number, int64_t target_number_hint) {
  if (target_number_hint > 0 && target_number_hint < magic_number) {
    return target_number_hint;
  }
  int64_t default_target_number = GetDefaultTargetNumber(magic_number);
  ABSL_LOG(WARNING) << "Target number hint " << target_number_hint
                    << " is larger than magic number " << magic_number
                    << ". Use default target number " << default_target_number;
  return default_target_number;
}

}  // namespace

std::vector<Environment::Option> MagicNumberConfigsHelper::GetLiteRtEnvOptions(
    const Model& litert_model, const LlmExecutorSettings& executor_settings) {
  auto magic_numbers = GetMagicNumbersFromModel(litert_model);
  if (!magic_numbers || (magic_numbers->context_length == 0 &&
                         magic_numbers->prefill_lengths.empty())) {
    return {};
  }

  // Build magic number configs.
  int prefill_config_index_base = magic_numbers->context_length > 0 ? 1 : 0;
  int num_configs =
      prefill_config_index_base + magic_numbers->prefill_lengths.size();
  magic_number_configs_ = UniqueCPtr<LiteRtMagicNumberConfigs>(
      reinterpret_cast<LiteRtMagicNumberConfigs*>(
          malloc(sizeof(LiteRtMagicNumberConfigs) +
                 num_configs * sizeof(LiteRtMagicNumberConfig))));
  magic_number_configs_->num_configs = num_configs;

  MagicNumbers target_numbers{.context_length = 0};
  // Magic number configs for context length.
  if (magic_numbers->context_length != 0) {
    auto& config = magic_number_configs_->configs[0];
    config.magic_number = magic_numbers->context_length;
    config.target_number = target_numbers.context_length = GetTargetNumber(
        config.magic_number, executor_settings.GetMaxNumTokens());
    config.signature_prefix = nullptr;
  }

  AdvancedSettings advanced_settings;
  if (executor_settings.GetAdvancedSettings()) {
    advanced_settings = *executor_settings.GetAdvancedSettings();
  }

  // How many extra magic numbers are there. If > 0, we can try to match the
  // magic number as close to the target number as possible.
  // If < 0, we have too many target numbers and skip first N target numbers.
  int extra_magic_numbers = magic_numbers->prefill_lengths.size() -
                            advanced_settings.prefill_batch_sizes.size();
  if (extra_magic_numbers < 0) {
    ABSL_LOG(WARNING) << "Too many prefill batch sizes="
                      << advanced_settings.prefill_batch_sizes.size()
                      << " for magic numbers of prefill lengths="
                      << magic_numbers->prefill_lengths.size() << ". "
                      << -extra_magic_numbers
                      << " small prefill batch sizes won't be used.";
  }

  if (!magic_numbers->prefill_lengths.empty()) {
    // Magic number configs for prefill length. Try to match with the smallest
    // magic number greater than or equal to the target number. Log warning if
    // there is no such magic numbers, but don't fail.
    auto it_prefill = advanced_settings.prefill_batch_sizes.begin();
    // Skip first -extra_magic_numbers target numbers.
    for (; extra_magic_numbers < 0; ++extra_magic_numbers) {
      ++it_prefill;
    }

    for (auto magic_number : magic_numbers->prefill_lengths) {
      // Fill with default target numbers until the magic number is larger than
      // or equal to the prefill batch size.
      int64_t target_prefill_number = 0;
      if (it_prefill == advanced_settings.prefill_batch_sizes.end() ||
          *it_prefill > magic_number) {
        target_prefill_number = GetDefaultTargetNumber(magic_number);
      } else {
        target_prefill_number = *it_prefill;
        ++it_prefill;
      }
      // Target prefill length must not be longer than context length.
      if (target_numbers.context_length > 0 &&
          target_prefill_number > target_numbers.context_length) {
        target_prefill_number = target_numbers.context_length;
      }
      target_numbers.prefill_lengths.push_back(target_prefill_number);
    }

    if (it_prefill != advanced_settings.prefill_batch_sizes.end()) {
      auto num_left = std::distance(
          it_prefill, advanced_settings.prefill_batch_sizes.end());
      ABSL_LOG(WARNING) << "No magic numbers for prefill length larger than or "
                        << "equal to the prefill batch size=" << *it_prefill
                        << ". Ignore last " << num_left
                        << " prefill batch sizes.";
    }

    for (int i = 0; i < magic_numbers->prefill_lengths.size(); ++i) {
      auto& config =
          magic_number_configs_->configs[prefill_config_index_base + i];
      config.magic_number = magic_numbers->prefill_lengths[i];
      config.target_number = target_numbers.prefill_lengths[i];
      config.signature_prefix = kPrefillSignaturePrefix.data();
    }
  }

  if (advanced_settings.verify_magic_numbers) {
    auto verify_pairs =
        GetVerificationPairs(litert_model, *magic_numbers, target_numbers);
    if (verify_pairs && !verify_pairs->empty()) {
      magic_number_verifications_ = UniqueCPtr<LiteRtMagicNumberVerifications>(
          reinterpret_cast<LiteRtMagicNumberVerifications*>(malloc(
              sizeof(LiteRtMagicNumberVerifications) +
              verify_pairs->size() * sizeof(LiteRtMagicNumberVerification))));
      magic_number_verifications_->num_verifications = verify_pairs->size();
      for (int i = 0; i < verify_pairs->size(); ++i) {
        auto& verification = magic_number_verifications_->verifications[i];
        verification.signature = verify_pairs->at(i).first.data();
        verification.test_signature = verify_pairs->at(i).second.data();
        verification.is_superset = true;
      }
    }
  }

  std::vector<Environment::Option> env_options;
  env_options.push_back(
      Environment::Option{Environment::OptionTag::MagicNumberConfigs,
                          static_cast<void*>(magic_number_configs_.get())});

  if (magic_number_verifications_) {
    env_options.push_back(Environment::Option{
        Environment::OptionTag::MagicNumberVerifications,
        static_cast<void*>(magic_number_verifications_.get())});
  }

  return env_options;
}

}  // namespace litert::lm
