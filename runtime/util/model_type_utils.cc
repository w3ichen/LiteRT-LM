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

#include "runtime/util/model_type_utils.h"

#include <array>
#include <string>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/substitute.h"  // from @com_google_absl
#include "runtime/components/tokenizer.h"
#include "runtime/proto/llm_metadata.pb.h"
#include "runtime/proto/llm_model_type.pb.h"
#include "runtime/proto/token.pb.h"
#include "runtime/util/status_macros.h"  // IWYU pragma: keep

namespace litert::lm {
namespace {

constexpr std::array<int, 1> kStartTurnTokenIdsToCheck = {
    105,  // Gemma family.
};

}  // namespace

absl::StatusOr<proto::LlmModelType> InferLlmModelType(
    const proto::LlmMetadata& metadata, Tokenizer& tokenizer) {
  proto::LlmModelType model_type;
  model_type.mutable_generic_model();
  for (int token_id : kStartTurnTokenIdsToCheck) {
    ASSIGN_OR_RETURN(auto start_turn_text,
                     tokenizer.TokenIdsToText({token_id}));
    // Gemma family.
    if (start_turn_text == "<start_of_turn>") {
      ASSIGN_OR_RETURN(auto audio_token_ids,
                       tokenizer.TextToTokenIds("<start_of_audio>"));
      if (audio_token_ids.size() == 1 && audio_token_ids[0] == 256000) {
        model_type.mutable_gemma3n()
            ->mutable_start_of_image_token()
            ->set_token_str("<start_of_image>");
        model_type.mutable_gemma3n()
            ->mutable_end_of_image_token()
            ->set_token_str("<end_of_image>");
        model_type.mutable_gemma3n()->set_image_tensor_height(768);
        model_type.mutable_gemma3n()->set_image_tensor_width(768);
        model_type.mutable_gemma3n()
            ->mutable_start_of_audio_token()
            ->set_token_str("<start_of_audio>");
        model_type.mutable_gemma3n()
            ->mutable_end_of_audio_token()
            ->set_token_str("<end_of_audio>");
        break;
      } else {
        // LiteRT-LM only supports Gemma3 1B model which doesn't have audio
        // tokens in the tokenizer.
        model_type.mutable_gemma3();
        break;
      }
    }
  }
  return model_type;
}

absl::StatusOr<std::string> GetDefaultJinjaPromptTemplate(
    const proto::PromptTemplates& prompt_templates,
    const proto::LlmModelType& llm_model_type) {
  switch (llm_model_type.model_type_case()) {
    case proto::LlmModelType::kGemma3N:
    case proto::LlmModelType::kGemma3:
    case proto::LlmModelType::kGenericModel:
      // absl::Substitute takes up to 10 arguments, so we have to split the
      // template into two parts.
      return absl::StrCat(
          absl::Substitute("{%- for message in messages -%}"
                           "{%- if message.content is string -%}"
                           "{%- if message.role == 'user' %}"
                           "$0{{ message.content }}$1"
                           "{% endif -%}"
                           "{%- if message.role == 'model' %}"
                           "$2{{ message.content }}$3"
                           "{% endif -%}"
                           "{%- if message.role == 'system' %}"
                           "$4{{ message.content }}$5"
                           "{% endif -%}"
                           "{%- else -%}",
                           prompt_templates.user().prefix(),
                           prompt_templates.user().suffix(),
                           prompt_templates.model().prefix(),
                           prompt_templates.model().suffix(),
                           prompt_templates.system().prefix(),
                           prompt_templates.system().suffix()),
          absl::Substitute("{%- if message.role == 'user' %}"
                           "$0"
                           "{% elif message.role == 'model' %}"
                           "$1"
                           "{% elif message.role == 'system' %}"
                           "$2"
                           "{% endif -%}"
                           "{%- for item in message.content %}"
                           "{%- if item.type == 'text' %}"
                           "{{ item.text }}"
                           "{% elif item.type == 'image' -%}"
                           "{{ '<start_of_image>' }}"
                           "{%- elif item.type == 'audio' -%}"
                           "{{ '<start_of_audio>' }}"
                           "{%- endif -%}"
                           "{%- endfor -%}"
                           "{%- if message.role == 'user' %}"
                           "$3"
                           "{% elif message.role == 'model' %}"
                           "$4"
                           "{% elif message.role == 'system' %}"
                           "$5"
                           "{% endif -%}"
                           "{%- endif -%}"
                           "{%- endfor -%}"
                           "{%- if add_generation_prompt %}"
                           "$6"
                           "{% endif -%}",
                           prompt_templates.user().prefix(),
                           prompt_templates.model().prefix(),
                           prompt_templates.system().prefix(),
                           prompt_templates.user().suffix(),
                           prompt_templates.model().suffix(),
                           prompt_templates.system().suffix(),
                           prompt_templates.model().prefix()));
    case proto::LlmModelType::MODEL_TYPE_NOT_SET:
      return absl::InvalidArgumentError("LlmModelType is not set.");
  }
}

}  // namespace litert::lm
