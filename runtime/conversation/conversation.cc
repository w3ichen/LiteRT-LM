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

#include "runtime/conversation/conversation.h"

#include <cstdint>
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
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "nlohmann/json.hpp"  // from @nlohmann_json
#include "runtime/components/prompt_template.h"
#include "runtime/conversation/internal_observable_adapter.h"
#include "runtime/conversation/io_types.h"
#include "runtime/conversation/model_data_processor/config_registry.h"
#include "runtime/conversation/model_data_processor/model_data_processor.h"
#include "runtime/conversation/model_data_processor/model_data_processor_factory.h"
#include "runtime/engine/engine.h"
#include "runtime/engine/io_types.h"
#include "runtime/proto/llm_model_type.pb.h"
#include "runtime/util/status_macros.h"

namespace litert::lm {

absl::StatusOr<std::string> Conversation::GetSingleTurnText(
    const Message& message) const {
  PromptTemplateInput old_tmpl_input;
  if (std::holds_alternative<JsonPreface>(preface_)) {
    auto json_preface = std::get<JsonPreface>(preface_);

    for (auto& message : json_preface.messages) {
      ASSIGN_OR_RETURN(nlohmann::ordered_json message_tmpl_input,
                       model_data_processor_->MessageToTemplateInput(message));
      old_tmpl_input.messages.push_back(message_tmpl_input);
    }

    if (json_preface.tools.is_null()) {
      old_tmpl_input.tools = nullptr;
    } else {
      ASSIGN_OR_RETURN(old_tmpl_input.tools,
                       model_data_processor_->FormatTools(json_preface.tools));
    }
    old_tmpl_input.extra_context = json_preface.extra_context;
  } else {
    return absl::UnimplementedError("Preface type is not supported yet");
  }
  absl::MutexLock lock(&history_mutex_);  // NOLINT
  for (const auto& history_msg : history_) {
    if (std::holds_alternative<nlohmann::ordered_json>(history_msg)) {
      ASSIGN_OR_RETURN(nlohmann::ordered_json message_tmpl_input,
                       model_data_processor_->MessageToTemplateInput(
                           std::get<nlohmann::ordered_json>(history_msg)));
      old_tmpl_input.messages.push_back(message_tmpl_input);
    } else {
      return absl::UnimplementedError("Message type is not supported yet");
    }
  }

  if (history_.empty()) {
    PromptTemplateInput new_tmpl_input = std::move(old_tmpl_input);
    if (std::holds_alternative<nlohmann::ordered_json>(message)) {
      ASSIGN_OR_RETURN(nlohmann::ordered_json message_tmpl_input,
                       model_data_processor_->MessageToTemplateInput(
                           std::get<nlohmann::ordered_json>(message)));
      new_tmpl_input.messages.push_back(message_tmpl_input);
    } else {
      return absl::UnimplementedError("Message type is not supported yet");
    }
    new_tmpl_input.add_generation_prompt = true;
    return prompt_template_.Apply(new_tmpl_input);
  }

  old_tmpl_input.add_generation_prompt = false;
  ASSIGN_OR_RETURN(const std::string old_string,
                   prompt_template_.Apply(old_tmpl_input));

  if (std::holds_alternative<nlohmann::ordered_json>(message)) {
    PromptTemplateInput new_tmpl_input = std::move(old_tmpl_input);
    ASSIGN_OR_RETURN(nlohmann::ordered_json message_tmpl_input,
                     model_data_processor_->MessageToTemplateInput(
                         std::get<nlohmann::ordered_json>(message)));
    new_tmpl_input.messages.push_back(message_tmpl_input);
    new_tmpl_input.add_generation_prompt = true;
    ASSIGN_OR_RETURN(const std::string& new_string,
                     prompt_template_.Apply(new_tmpl_input));
    if (new_string.substr(0, old_string.size()) != old_string) {
      return absl::InternalError(absl::StrCat(
          "The new rendered template string does not start with the previous "
          "rendered template string. \nold_string: ",
          old_string, "\nnew_string: ", new_string));
    }
    return {new_string.substr(old_string.size(),
                              new_string.size() - old_string.size())};
  } else {
    return absl::InvalidArgumentError("Json message is required for now.");
  }
}

absl::StatusOr<std::unique_ptr<Conversation>> Conversation::Create(
    std::unique_ptr<Engine::Session> session, std::optional<Preface> preface,
    std::optional<PromptTemplate> prompt_template,
    std::optional<DataProcessorConfig> processor_config) {
  if (!preface.has_value()) {
    preface = JsonPreface();
  }
  const proto::LlmModelType& llm_model_type =
      session->GetSessionConfig().GetLlmModelType();
  processor_config = processor_config.value_or(std::monostate());
  ASSIGN_OR_RETURN(
      std::unique_ptr<ModelDataProcessor> model_data_processor,
      CreateModelDataProcessor(llm_model_type, *processor_config, *preface));
  if (!prompt_template.has_value()) {
    // Get template from the session or model file when the template is not
    // provided by the user.
    ABSL_LOG(INFO)
        << "Prompt template is not provided, using default template.";
    prompt_template =
        PromptTemplate(session->GetSessionConfig().GetJinjaPromptTemplate());
  }
  auto conversation = absl::WrapUnique(
      new Conversation(std::move(session), std::move(model_data_processor),
                       *preface, *prompt_template));
  return conversation;
}

absl::StatusOr<Message> Conversation::SendMessage(
    const Message& message, std::optional<DataProcessorArguments> args) {
  if (!std::holds_alternative<nlohmann::ordered_json>(message)) {
    return absl::InvalidArgumentError("Json message is required for now.");
  }
  auto json_message = std::get<nlohmann::ordered_json>(message);
  ASSIGN_OR_RETURN(const std::string& single_turn_text,
                   GetSingleTurnText(message));
  absl::MutexLock lock(&history_mutex_);  // NOLINT
  history_.push_back(json_message);
  ASSIGN_OR_RETURN(
      const auto session_inputs,
      model_data_processor_->ToInputDataVector(
          single_turn_text, nlohmann::ordered_json::array({json_message}),
          args.value_or(std::monostate())));
  ASSIGN_OR_RETURN(const Responses& responses,
                   session_->GenerateContent(session_inputs));
  ASSIGN_OR_RETURN(const Message assistant_message,
                   model_data_processor_->ToMessage(
                       responses, args.value_or(std::monostate())));
  history_.push_back(assistant_message);
  return assistant_message;
}

absl::Status Conversation::SendMessageStream(
    const Message& message, MessageObservable* observer,
    std::optional<DataProcessorArguments> args) {
  if (!std::holds_alternative<nlohmann::ordered_json>(message)) {
    return absl::InvalidArgumentError("Json message is required for now.");
  }
  auto json_message = std::get<nlohmann::ordered_json>(message);
  ASSIGN_OR_RETURN(const std::string& single_turn_text,
                   GetSingleTurnText(message));
  {
    absl::MutexLock lock(&history_mutex_);  // NOLINT
    history_.push_back(message);
  }

  ASSIGN_OR_RETURN(
      const auto session_inputs,
      model_data_processor_->ToInputDataVector(
          single_turn_text, nlohmann::ordered_json::array({json_message}),
          args.value_or(std::monostate())));

  auto internal_observable_adapter = InternalObservableAdapter::Create(
      model_data_processor_.get(), observer, args.value_or(std::monostate()));

  InternalObservableAdapter::CompleteMessageCallback complete_message_callback =
      [&internal_observable_adapter, this](const Message& complete_message) {
        absl::MutexLock lock(&this->history_mutex_);  // NOLINT
        this->history_.push_back(complete_message);
        this->observable_map_.erase(
            reinterpret_cast<uintptr_t>(internal_observable_adapter.get()));
      };
  internal_observable_adapter->SetCompleteMessageCallback(
      std::move(complete_message_callback));

  auto internal_observable_adapter_ptr = internal_observable_adapter.get();
  observable_map_[reinterpret_cast<uintptr_t>(
      internal_observable_adapter_ptr)] =
      std::move(internal_observable_adapter);
  RETURN_IF_ERROR(session_->RunPrefill(session_inputs));
  RETURN_IF_ERROR(session_->RunDecodeAsync(internal_observable_adapter_ptr));
  return absl::OkStatus();
};

}  // namespace litert::lm
