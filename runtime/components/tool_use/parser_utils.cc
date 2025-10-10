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
#include "runtime/util/status_macros.h"
#include "re2/re2.h"  // from @com_googlesource_code_re2

namespace litert::lm {

namespace {

RE2 TextAndToolCodeRegex(absl::string_view code_fence_start,
                         absl::string_view code_fence_end,
                         bool escape_fence_strings) {
  // Construct the regex pattern: (non-greedy text before) <start> (non-greedy
  // code) <end>.
  std::string pattern;
  if (escape_fence_strings) {
    // QuoteMeta escapes any special regex characters in the fence strings.
    pattern = absl::StrCat("(?ms)(.*?)", RE2::QuoteMeta(code_fence_start),
                           "(.*?)", RE2::QuoteMeta(code_fence_end));
  } else {
    pattern =
        absl::StrCat("(?ms)(.*?)", code_fence_start, "(.*?)", code_fence_end);
  }
  return RE2(pattern);
}

std::string FilterLines(absl::string_view input, const RE2& regex) {
  std::vector<absl::string_view> lines = absl::StrSplit(input, '\n');
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

absl::StatusOr<nlohmann::ordered_json> ParseTextAndToolCalls(
    absl::string_view response_str, absl::string_view code_fence_start,
    absl::string_view code_fence_end, SyntaxType syntax_type,
    bool escape_fence_strings, absl::string_view tool_code_regex) {
  nlohmann::ordered_json result = nlohmann::json::object();
  RE2 regex = TextAndToolCodeRegex(code_fence_start, code_fence_end,
                                   escape_fence_strings);
  if (!regex.ok()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Invalid regex: ", regex.pattern(), " error: ", regex.error()));
  }

  std::string text;
  std::string code_block;
  while (RE2::Consume(&response_str, regex, &text, &code_block)) {
    // Append text to the content array.
    if (!text.empty()) {
      result["content"].push_back({{"type", "text"}, {"text", text}});
    }

    // Before parsing the code block, apply tool_code_regex to each line.
    if (!tool_code_regex.empty()) {
      RE2 regex(tool_code_regex);
      if (!regex.ok()) {
        return absl::InvalidArgumentError(
            absl::StrCat("Invalid tool_code_regex: ", tool_code_regex));
      }
      code_block = FilterLines(code_block, regex);
    }

    // Parse tool calls from the code block.
    if (!code_block.empty()) {
      nlohmann::ordered_json tool_calls;
      if (syntax_type == SyntaxType::kPython) {
        ASSIGN_OR_RETURN(tool_calls, ParsePythonExpression(code_block));
      } else if (syntax_type == SyntaxType::kJson) {
        ASSIGN_OR_RETURN(tool_calls, ParseJsonExpression(code_block));
      } else {
        return absl::InvalidArgumentError("Unsupported syntax type.");
      }
      for (const auto& tool_call : tool_calls) {
        result["tool_calls"].push_back(
            {{"type", "function"}, {"function", tool_call}});
      }
    }
    text.clear();
    code_block.clear();
  }

  // Append the remaining text to the content array.
  if (!response_str.empty()) {
    result["content"].push_back({{"type", "text"}, {"text", response_str}});
  }

  return result;
}

}  // namespace litert::lm
