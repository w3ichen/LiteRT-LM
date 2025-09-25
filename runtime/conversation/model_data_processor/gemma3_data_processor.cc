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

#include "runtime/conversation/model_data_processor/gemma3_data_processor.h"

#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "nlohmann/json.hpp"  // from @nlohmann_json
#include "runtime/components/tool_use/parser_utils.h"
#include "runtime/components/tool_use/python_tool_format_utils.h"
#include "runtime/conversation/io_types.h"
#include "runtime/conversation/model_data_processor/gemma3_data_processor_config.h"
#include "runtime/engine/io_types.h"
#include "runtime/util/status_macros.h"

namespace litert::lm {

absl::StatusOr<std::unique_ptr<Gemma3DataProcessor>>
Gemma3DataProcessor::Create(Gemma3DataProcessorConfig config,
                            std::optional<Preface> preface) {
  return absl::WrapUnique(new Gemma3DataProcessor(config, preface));
}

absl::StatusOr<std::vector<InputData>>
Gemma3DataProcessor::ToInputDataVectorImpl(
    const std::string& rendered_template_prompt,
    const nlohmann::ordered_json& messages,
    const Gemma3DataProcessorArguments& args) {
  std::vector<InputData> input_data;
  int current_pos = 0;
  while (current_pos < rendered_template_prompt.length()) {
    int boi_pos = rendered_template_prompt.find(config_.boi_token, current_pos);
    if (boi_pos == std::string::npos) {
      input_data.push_back(
          InputText(rendered_template_prompt.substr(current_pos)));
      break;
    }

    // Add text before the image token.
    if (boi_pos > current_pos) {
      input_data.push_back(InputText(
          rendered_template_prompt.substr(current_pos, boi_pos - current_pos)));
      // TODO: b/438830175 - Add preprocessed image tensor once image
      // preprocessor is ready.
      input_data.push_back(InputImage(""));
    }
    current_pos = boi_pos + config_.boi_token.length();
  }
  return input_data;
}

absl::StatusOr<Message> Gemma3DataProcessor::ToMessageImpl(
    const Responses& responses, const Gemma3DataProcessorArguments& args) {
  ASSIGN_OR_RETURN(absl::string_view response_text,
                   responses.GetResponseTextAt(0));
  nlohmann::ordered_json content;
  if (preface_.has_value() && std::holds_alternative<JsonPreface>(*preface_) &&
      !std::get<JsonPreface>(*preface_).tools.empty()) {
    ASSIGN_OR_RETURN(
        content, ParseTextAndToolCalls(
                     response_text, config_.code_fence_start,
                     config_.code_fence_end, GetSyntaxType(config_.syntax_type),
                     config_.escape_fence_strings, config_.tool_code_regex));
  } else {
    content = nlohmann::ordered_json::array(
        {{{"type", "text"}, {"text", std::string(response_text)}}});
  }
  return nlohmann::ordered_json::object(
      {{"role", "assistant"}, {"content", content}});
}

absl::StatusOr<nlohmann::ordered_json> Gemma3DataProcessor::FormatTools(
    const nlohmann::ordered_json& tools) {
  if (!tools.is_array()) {
    return absl::InvalidArgumentError("Tools must be an array.");
  }
  nlohmann::ordered_json formatted_tools = nlohmann::ordered_json::array();
  for (const auto& tool : tools) {
    ASSIGN_OR_RETURN(std::string formatted_tool, FormatToolAsPython(tool));
    formatted_tools.push_back(formatted_tool);
  }
  return formatted_tools;
}

absl::string_view Gemma3DataProcessor::CodeFenceStart() {
  return config_.code_fence_start;
}

absl::string_view Gemma3DataProcessor::CodeFenceEnd() {
  return config_.code_fence_end;
}

}  // namespace litert::lm
