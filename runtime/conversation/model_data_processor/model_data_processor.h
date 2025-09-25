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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CONVERSATION_MODEL_DATA_PROCESSOR_MODEL_DATA_PROCESSOR_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CONVERSATION_MODEL_DATA_PROCESSOR_MODEL_DATA_PROCESSOR_H_

#include <string>
#include <variant>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "nlohmann/json.hpp"  // from @nlohmann_json
#include "runtime/conversation/io_types.h"
#include "runtime/conversation/model_data_processor/config_registry.h"
#include "runtime/engine/io_types.h"

namespace litert::lm {

// ModelDataProcessor is a model-specific component that converts between the
// generic Json messages and the Litert LM InputData type.
class ModelDataProcessor {
 public:
  virtual ~ModelDataProcessor() = default;

  // Converts a rendered template prompt and a list of messages to a vector of
  // InputData, which is the input to the LLM Session.
  virtual absl::StatusOr<std::vector<InputData>> ToInputDataVector(
      const std::string& rendered_template_prompt,
      const nlohmann::ordered_json& messages,
      const DataProcessorArguments& args) = 0;

  // Converts a list of responses from the LLM Session to a Message, which is
  // the output to the user.
  virtual absl::StatusOr<Message> ToMessage(
      const Responses& responses, const DataProcessorArguments& args) = 0;

  // Formats the provided tools to be inserted into the system/developer
  // instruction of the prompt.
  virtual absl::StatusOr<nlohmann::ordered_json> FormatTools(
      const nlohmann::ordered_json& tools) = 0;

  // Returns the start of tool call blocks.
  virtual absl::string_view CodeFenceStart() = 0;

  // Returns the end of tool call blocks.
  virtual absl::string_view CodeFenceEnd() = 0;
};

// TypeSafeModelDataProcessor is a ModelDataProcessor that expects a specific
// type of arguments. It guarantees that the model data processor will only be
// called with the expected arguments type.
//
// The model data processor should overwrite the ToInputDataVectorImpl and
// ToMessageImpl to handle the model-specific logic.
template <typename ExpectedConfigT, typename ExpectedArgsT>
class TypeSafeModelDataProcessor : public ModelDataProcessor {
 public:
  // Converts a rendered template prompt and a list of messages to a vector of
  // InputData, with arguments type validated.
  absl::StatusOr<std::vector<InputData>> ToInputDataVector(
      const std::string& rendered_template_prompt,
      const nlohmann::ordered_json& messages,
      const DataProcessorArguments& args) final {
    if (std::holds_alternative<ExpectedArgsT>(args)) {
      return this->ToInputDataVectorImpl(rendered_template_prompt, messages,
                                         std::get<ExpectedArgsT>(args));
    } else if (std::holds_alternative<std::monostate>(args)) {
      return this->ToInputDataVectorImpl(rendered_template_prompt, messages,
                                         ExpectedArgsT{});
    }
    return absl::InvalidArgumentError(
        "DataProcessorArguments does not hold the expected type");
  }

  // Converts a list of responses from the LLM Session to a Message, with
  // arguments type validated.
  absl::StatusOr<Message> ToMessage(const Responses& responses,
                                    const DataProcessorArguments& args) final {
    if (std::holds_alternative<ExpectedArgsT>(args)) {
      return this->ToMessageImpl(responses, std::get<ExpectedArgsT>(args));
    } else if (std::holds_alternative<std::monostate>(args)) {
      return this->ToMessageImpl(responses, ExpectedArgsT{});
    }
    return absl::InvalidArgumentError(
        "DataProcessorArguments does not hold the expected type");
  }

  // Returns the config of the model data processor.
  virtual const ExpectedConfigT& GetConfig() = 0;

 private:
  virtual absl::StatusOr<std::vector<InputData>> ToInputDataVectorImpl(
      const std::string& rendered_template_prompt,
      const nlohmann::ordered_json& messages,
      const ExpectedArgsT& typed_args) = 0;

  virtual absl::StatusOr<Message> ToMessageImpl(
      const Responses& responses, const ExpectedArgsT& typed_args) = 0;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CONVERSATION_MODEL_DATA_PROCESSOR_MODEL_DATA_PROCESSOR_H_
