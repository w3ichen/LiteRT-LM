// Copyright 2025 The Google AI Edge Authors.
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

#include "runtime/components/tool_use/parser_utils.h"

#include <string>
#include <vector>

#include "absl/base/no_destructor.h"  // from @com_google_absl
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_join.h"  // from @com_google_absl
#include "absl/strings/str_split.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "nlohmann/json.hpp"  // from @nlohmann_json
#include "runtime/components/tool_use/json_parser_utils.h"
#include "runtime/components/tool_use/python_parser_utils.h"
#include "re2/re2.h"  // from @com_googlesource_code_re2

namespace litert::lm {

namespace {

std::string FilterToolCallString(const std::string& tool_call_string,
                                 const RE2& regex) {
  std::vector<absl::string_view> lines = absl::StrSplit(tool_call_string, '\n');
  std::string captured_part;
  std::vector<std::string> captured_lines;

  for (absl::string_view line : lines) {
    if (RE2::PartialMatch(line, regex, &captured_part)) {
      captured_lines.push_back(captured_part);
    } else {
      captured_lines.push_back(std::string(line));
    }
  }

  return absl::StrJoin(captured_lines, "\n");
}

}  // namespace

SyntaxType GetSyntaxType(absl::string_view syntax_type) {
  static const absl::NoDestructor<
      absl::flat_hash_map<absl::string_view, SyntaxType>>
      kStringToSyntaxType({
          {"python", SyntaxType::kPython},
          {"json", SyntaxType::kJson},
      });
  auto it = kStringToSyntaxType->find(syntax_type);
  if (it == kStringToSyntaxType->end()) {
    return SyntaxType::kUnknown;
  }
  return it->second;
}

TextAndToolCallStrings ExtractTextAndToolCallStrings(
    absl::string_view response_str, absl::string_view code_fence_start,
    absl::string_view code_fence_end, bool escape_fence_strings) {
  TextAndToolCallStrings result;
  absl::string_view text_before;
  absl::string_view code_block;

  // Construct the regex pattern: (non-greedy text before) <start> (non-greedy
  // code) <end> QuoteMeta escapes any special regex characters in the fence
  // strings.
  std::string pattern;
  if (escape_fence_strings) {
    pattern = absl::StrCat("(?ms)(.*?)", RE2::QuoteMeta(code_fence_start),
                           "(.*?)", RE2::QuoteMeta(code_fence_end));
  } else {
    pattern =
        absl::StrCat("(?ms)(.*?)", code_fence_start, "(.*?)", code_fence_end);
  }
  RE2 regex(pattern);
  if (RE2::PartialMatch(response_str, regex, &text_before, &code_block)) {
    // Found both start and end fences.
    result.text = text_before;
    result.tool_calls = code_block;
  } else {
    // If both code fences are not found, treat the entire string as text.
    result.text = response_str;
  }
  return result;
}

absl::StatusOr<nlohmann::ordered_json> ParseTextAndToolCalls(
    absl::string_view response_str, absl::string_view code_fence_start,
    absl::string_view code_fence_end, SyntaxType syntax_type,
    bool escape_fence_strings, absl::string_view tool_code_regex) {
  nlohmann::ordered_json content = nlohmann::json::array();
  TextAndToolCallStrings text_and_tool_call_strings =
      ExtractTextAndToolCallStrings(response_str, code_fence_start,
                                    code_fence_end, escape_fence_strings);
  if (!text_and_tool_call_strings.text.empty()) {
    content.push_back(
        {{"type", "text"}, {"text", text_and_tool_call_strings.text}});
  }
  if (!text_and_tool_call_strings.tool_calls.empty()) {
    std::string tool_calls_to_parse =
        std::string(text_and_tool_call_strings.tool_calls);
    if (!tool_code_regex.empty()) {
      RE2 regex(tool_code_regex);
      if (!regex.ok()) {
        return absl::InvalidArgumentError(
            absl::StrCat("Invalid tool_code_regex: ", tool_code_regex));
      }
      tool_calls_to_parse = FilterToolCallString(tool_calls_to_parse, regex);
      if (tool_calls_to_parse.empty()) {
        return content;
      }
    }

    absl::StatusOr<nlohmann::ordered_json> tool_calls;
    if (syntax_type == SyntaxType::kPython) {
      tool_calls = ParsePythonExpression(tool_calls_to_parse);
    } else if (syntax_type == SyntaxType::kJson) {
      tool_calls = ParseJsonExpression(tool_calls_to_parse);
    } else {
      return absl::InvalidArgumentError("Unsupported syntax type.");
    }
    if (tool_calls.ok()) {
      for (const auto& tool_call : *tool_calls) {
        nlohmann::ordered_json content_part;
        content_part["type"] = "tool_call";
        content_part["tool_call"] = tool_call;
        content.push_back(content_part);
      }
    } else {
      return absl::InternalError("Failed to parse tool call from output.");
    }
  }
  return content;
}

}  // namespace litert::lm
