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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CONVERSATION_CONVERSATION_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CONVERSATION_CONVERSATION_H_

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "runtime/components/prompt_template.h"
#include "runtime/conversation/io_types.h"
#include "runtime/conversation/model_data_processor/config_registry.h"
#include "runtime/conversation/model_data_processor/model_data_processor.h"
#include "runtime/engine/engine.h"
#include "runtime/engine/io_types.h"

namespace litert::lm {

// A multi-turn centric stateful Conversation API for high-level user
// interaction. Conversation maintains the history for users, so the users'
// messages will be used as the LLM context through the conversation.
//
// Conversation handles the complex data processing logic for Session usage,
// including:
// - Prompt template rendering.
// - Role-based messages handling.
// - Multimodal input processing.
// - History management.
// - Model-specific data processing.
//
// Example usage:
//
//   // Create a Conversation instance, by transferring the ownership of the
//   // Session instance to the Conversation. The Session instance is created
//   // from the LLM Engine, and Conversation acts as a delegate for users to
//   // interact with the LLM Session.
//   ASSIGN_OR_RETURN(auto conversation,
//       Conversation::Create(std::move(session)));
//
//   // Send a message to the LLM and returns the complete message.
//   ASSIGN_OR_RETURN(const Message message,
//                    conversation->SendMessage(JsonMessage{
//                        {"role", "user"}, {"content", "Hello world!"}}));
//
//   // Send a message to the LLM and process the asynchronous message results
//   // via the callbacks.
//   // The callbacks is a user-defined callback class that handles the message
//   // results.
//   MyMessageObservable my_message_observable();
//   EXPECT_OK(conversation->SendMessageStream(
//       JsonMessage{{"role", "user"}, {"content", "Hello world!"}},
//       &my_message_observable));
//
class Conversation {
 public:
  // Creates a Conversation instance.
  // Args:
  // - `session`: The Session instance to be used for the conversation.
  // - `preface`: Optional Preface for the conversation. The Preface provides
  //     the initial background for the conversation, tool uses and extra
  //     context for the conversation. If not provided, the conversation will
  //     start with an empty Preface.
  // - `prompt_template`: Optional PromptTemplate instance to be used for the
  //     conversation. If not provided, the conversation will use the default
  //     template for the model.
  // - `processor_config`: Optional configuration for the model data processor,
  //    if not provided, the default model config for data processing will be
  //    used. Most of the time, the users don't need to provide the config.
  static absl::StatusOr<std::unique_ptr<Conversation>> Create(
      std::unique_ptr<Engine::Session> session,
      std::optional<Preface> preface = std::nullopt,
      std::optional<PromptTemplate> prompt_template = std::nullopt,
      std::optional<DataProcessorConfig> processor_config = std::nullopt);

  // Sends a message to the LLM and returns the complete message.
  // Args:
  // - `message`: The message to be sent to the LLM.
  // - `args`: The optional arguments for the corresponding model data
  //    processor. Most of the time, the users don't need to provide this
  //    argument.
  // Returns :
  // - The complete message from the LLM.
  absl::StatusOr<Message> SendMessage(
      const Message& message,
      std::optional<DataProcessorArguments> args = std::nullopt);

  // Sends a message to the LLM and process the asynchronous message results via
  // the callbacks.
  // Args:
  // - `message`: The message to be sent to the LLM.
  // - `callbacks`: The callbacks to receive the message events.
  // - `args`: The optional arguments for the corresponding model data
  //    processor. Most of the time, the users don't need to provide this
  //    argument.
  // Returns :
  // - absl::OkStatus if the message is sent and processing successfully,
  //   otherwise the error status.
  absl::Status SendMessageStream(
      const Message& message, std::unique_ptr<MessageCallbacks> callbacks,
      std::optional<DataProcessorArguments> args = std::nullopt);

  // Returns the history of the conversation.
  std::vector<Message> GetHistory() const {
    absl::MutexLock lock(&history_mutex_);  // NOLINT
    return history_;
  }

  // Returns the benchmark info for the conversation. Underlying this method
  // triggers the benchmark info collection from the Session.
  // Returns:
  // - The benchmark info for the conversation.
  absl::StatusOr<BenchmarkInfo> GetBenchmarkInfo();

  // Cancels the ongoing inference process, for asynchronous inference.
  // Note: the underlying Session is not rollbacked, so the message
  // from the user is actually sent to the LLM and processed for prefill.
  void CancelProcess();

 private:
  explicit Conversation(
      std::unique_ptr<Engine::Session> session,
      std::unique_ptr<ModelDataProcessor> model_data_processor, Preface preface,
      PromptTemplate prompt_template)
      : session_(std::move(session)),
        model_data_processor_(std::move(model_data_processor)),
        preface_(preface),
        prompt_template_(std::move(prompt_template)) {}

  absl::StatusOr<std::string> GetSingleTurnText(const Message& message) const;

  std::unique_ptr<Engine::Session> session_;
  std::unique_ptr<ModelDataProcessor> model_data_processor_;
  Preface preface_;
  PromptTemplate prompt_template_;
  mutable absl::Mutex history_mutex_;
  std::vector<Message> history_ ABSL_GUARDED_BY(history_mutex_);
};
}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CONVERSATION_CONVERSATION_H_
