// Copyright 2025 The ODML Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may
// may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "runtime/conversation/internal_observable_adapter.h"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <string>

#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "runtime/conversation/io_types.h"
#include "runtime/conversation/model_data_processor/config_registry.h"
#include "runtime/conversation/model_data_processor/model_data_processor.h"
#include "runtime/engine/io_types.h"

namespace litert::lm {
namespace {

// Returns the number of overlapping characters between the suffix of string `a`
// and the prefix of string `b`.
size_t SuffixPrefixOverlap(absl::string_view a, absl::string_view b) {
  if (a.empty() || b.empty()) {
    return false;
  }

  size_t max_overlap = std::min(a.length(), b.length());

  for (size_t len = max_overlap; len > 0; --len) {
    if (a.substr(a.length() - len) == b.substr(0, len)) {
      return len;
    }
  }

  return 0;
}

}  // namespace

std::unique_ptr<InternalObservableAdapter> InternalObservableAdapter::Create(
    ModelDataProcessor* model_data_processor, MessageObservable* user_observer,
    DataProcessorArguments processor_args) {
  return std::unique_ptr<InternalObservableAdapter>(
      new InternalObservableAdapter(model_data_processor, user_observer,
                                    processor_args));
}

void InternalObservableAdapter::SetCompleteMessageCallback(
    CompleteMessageCallback complete_message_callback) {
  complete_message_callback_ = complete_message_callback;
}

void InternalObservableAdapter::OnNext(const Responses& responses) {
  const auto& response_text = responses.GetResponseTextAt(0);
  if (!response_text.ok()) {
    user_observer_->OnError(response_text.status());
    return;
  }
  auto status = ProcessResponseText(*response_text);
  if (!status.ok()) {
    user_observer_->OnError(status);
    return;
  }
}

void InternalObservableAdapter::OnDone() {
  // Send the remaining text to the user observer when done.
  if (cursor_ < accumulated_response_text_.size()) {
    SendMessage(absl::string_view(accumulated_response_text_).substr(cursor_));
  }
  Responses responses(1);
  responses.GetMutableResponseTexts()[0] = accumulated_response_text_;
  const auto& complete_message =
      model_data_processor_->ToMessage(responses, processor_args_);
  if (!complete_message.ok()) {
    user_observer_->OnError(complete_message.status());
    return;
  }
  user_observer_->OnComplete();
  if (complete_message_callback_) {
    complete_message_callback_(*complete_message);
  }
  complete_message_callback_ = nullptr;
}

void InternalObservableAdapter::OnError(const absl::Status& status) {
  // TODO: b/435001805 - handle the max kv-cache size reached situation more
  // robustly.
  if (absl::StrContainsIgnoreCase(status.message(),
                                  "Maximum kv-cache size reached")) {
    ABSL_LOG(INFO) << "Maximum kv-cache size reached.";
    OnDone();
    return;
  }
  user_observer_->OnError(status);
}

InternalObservableAdapter::InternalObservableAdapter(
    ModelDataProcessor* model_data_processor, MessageObservable* user_observer,
    DataProcessorArguments processor_args)
    : model_data_processor_(model_data_processor),
      user_observer_(user_observer),
      processor_args_(processor_args) {}

void InternalObservableAdapter::SendMessage(absl::string_view text) {
  if (text.empty()) {
    return;
  }
  Responses responses(1);
  responses.GetMutableResponseTexts()[0] = text;
  const auto& message =
      model_data_processor_->ToMessage(responses, processor_args_);
  if (!message.ok()) {
    user_observer_->OnError(message.status());
    return;
  }
  user_observer_->OnMessage(*message);
}

absl::Status InternalObservableAdapter::ProcessResponseText(
    absl::string_view response_text) {
  accumulated_response_text_.append(response_text);
  if (model_data_processor_ == nullptr) {
    return absl::InvalidArgumentError(
        "model_data_processor_ must not be null.");
  }
  absl::string_view code_fence_start = model_data_processor_->CodeFenceStart();
  absl::string_view code_fence_end = model_data_processor_->CodeFenceEnd();

  while (cursor_ < accumulated_response_text_.size()) {
    if (!inside_tool_call_) {
      size_t code_fence_start_pos =
          accumulated_response_text_.find(code_fence_start, cursor_);
      if (code_fence_start_pos != std::string::npos) {
        // The text from the cursor up to the code fence is normal text.
        SendMessage(absl::string_view(accumulated_response_text_)
                        .substr(cursor_, code_fence_start_pos - cursor_));

        // Move cursor up to code_fence_start.
        cursor_ = code_fence_start_pos;
        inside_tool_call_ = true;
      } else {
        // code_fence_start not found, but we still need to check
        // if there's a partial match at the end of the string.
        size_t overlap = SuffixPrefixOverlap(
            absl::string_view(accumulated_response_text_).substr(cursor_),
            code_fence_start);

        if (overlap > 0) {
          // There's a partial match of the code fence at the end of the
          // string.
          size_t possible_start_pos =
              accumulated_response_text_.size() - overlap;

          // Call the callback with text up to the potential start of the
          // code fence.
          SendMessage(absl::string_view(accumulated_response_text_)
                          .substr(cursor_, possible_start_pos - cursor_));

          // Move cursor up to potential start of code fence.
          cursor_ = possible_start_pos;
          break;
        } else {
          // Remaining string is text.
          SendMessage(
              absl::string_view(accumulated_response_text_).substr(cursor_));
          cursor_ = accumulated_response_text_.size();
        }
      }
    }

    if (inside_tool_call_) {
      // Look for code fence end.
      size_t code_fence_end_pos = accumulated_response_text_.find(
          code_fence_end, cursor_ + code_fence_start.size());
      if (code_fence_end_pos != std::string::npos) {
        SendMessage(absl::string_view(accumulated_response_text_)
                        .substr(cursor_, code_fence_end_pos +
                                             code_fence_end.size() - cursor_));

        // Move cursor to end of tool code block.
        cursor_ = code_fence_end_pos + code_fence_end.size();
        inside_tool_call_ = false;
      } else {
        // We're inside a tool call but the code fence end has not been
        // found. Break for the next token.
        break;
      }
    }
  }

  return absl::OkStatus();
}

}  // namespace litert::lm
