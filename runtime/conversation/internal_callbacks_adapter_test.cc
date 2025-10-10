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

#include "runtime/conversation/internal_callbacks_adapter.h"

#include <memory>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "nlohmann/json.hpp"  // from @nlohmann_json
#include "runtime/conversation/io_types.h"
#include "runtime/conversation/model_data_processor/config_registry.h"
#include "runtime/conversation/model_data_processor/gemma3_data_processor.h"
#include "runtime/conversation/model_data_processor/gemma3_data_processor_config.h"
#include "runtime/engine/io_types.h"
#include "runtime/util/test_utils.h"  // NOLINT

namespace litert::lm {
namespace {

using ::testing::ElementsAre;
using ::testing::IsEmpty;
using ::testing::status::StatusIs;

Responses CreateResponses(absl::string_view response_text) {
  Responses responses(1);
  responses.GetMutableResponseTexts()[0] = response_text;
  return responses;
}

nlohmann::ordered_json TextMessage(absl::string_view text) {
  nlohmann::ordered_json message;
  message["role"] = "assistant";
  message["content"] = {{{"type", "text"}, {"text", text}}};
  return message;
}

class UserMessageCallbacks : public MessageCallbacks {
 public:
  UserMessageCallbacks(std::vector<nlohmann::ordered_json>& output, bool& done,
                       absl::Status& status)
      : output_(output), done_(done), status_(status) {}

  void OnMessage(const Message& message) override {
    output_.push_back(std::get<nlohmann::ordered_json>(message));
  }
  void OnComplete() override { done_ = true; }
  void OnError(const absl::Status& status) override {
    done_ = true;
    status_ = status;
  }

 private:
  std::vector<nlohmann::ordered_json>& output_;
  bool& done_;
  absl::Status& status_;
};

class InternalCallbacksAdapterTest : public testing::Test {
 protected:
  void SetUp() override {
    Gemma3DataProcessorConfig config;

    // Need a tool in the preface to trigger tool call parsing. The actual tool
    // definition is unimportant.
    JsonPreface preface{.tools = nlohmann::ordered_json::parse(R"json([{
                  "name": "tool_name",
                  "parameters": { "properties": { "x": { "type": "integer" } } }
                }])json")};
    ASSERT_OK_AND_ASSIGN(model_data_processor_,
                         Gemma3DataProcessor::Create(config, preface));

    processor_args_ = DataProcessorArguments();
  }

  std::unique_ptr<Gemma3DataProcessor> model_data_processor_;
  std::vector<nlohmann::ordered_json> output_;
  bool done_ = false;
  absl::Status status_;
  DataProcessorArguments processor_args_;
};

TEST_F(InternalCallbacksAdapterTest, OnDone) {
  auto callbacks = InternalCallbacksAdapter::Create(
      model_data_processor_.get(),
      std::make_unique<UserMessageCallbacks>(output_, done_, status_),
      processor_args_);

  callbacks->OnDone();

  EXPECT_THAT(output_, IsEmpty());
  EXPECT_TRUE(done_);
  EXPECT_OK(status_);
}

TEST_F(InternalCallbacksAdapterTest, OnError) {
  auto callbacks = InternalCallbacksAdapter::Create(
      model_data_processor_.get(),
      std::make_unique<UserMessageCallbacks>(output_, done_, status_),
      processor_args_);

  callbacks->OnError(absl::InternalError("error"));

  EXPECT_THAT(output_, IsEmpty());
  EXPECT_TRUE(done_);
  EXPECT_THAT(status_, StatusIs(absl::StatusCode::kInternal, "error"));
}

TEST_F(InternalCallbacksAdapterTest, Text) {
  auto callbacks = InternalCallbacksAdapter::Create(
      model_data_processor_.get(),
      std::make_unique<UserMessageCallbacks>(output_, done_, status_),
      processor_args_);

  callbacks->OnNext(CreateResponses("this "));
  callbacks->OnNext(CreateResponses("is "));
  callbacks->OnNext(CreateResponses("some "));
  callbacks->OnNext(CreateResponses("text"));

  EXPECT_THAT(output_, ElementsAre(TextMessage("this "), TextMessage("is "),
                                   TextMessage("some "), TextMessage("text")));
}

TEST_F(InternalCallbacksAdapterTest, ToolCall) {
  auto callbacks = InternalCallbacksAdapter::Create(
      model_data_processor_.get(),
      std::make_unique<UserMessageCallbacks>(output_, done_, status_),
      processor_args_);

  callbacks->OnNext(CreateResponses("```tool_code\n"));
  callbacks->OnNext(CreateResponses("tool_name"));
  callbacks->OnNext(CreateResponses("(x=1)"));
  callbacks->OnNext(CreateResponses("\n```"));

  EXPECT_THAT(output_, ElementsAre(nlohmann::ordered_json::parse(R"json({
                "role": "assistant",
                "tool_calls": [
                  {
                    "type": "function",
                    "function": {
                      "name": "tool_name",
                      "arguments": {
                        "x": 1
                      }
                    }
                  }
                ]
              })json")));
}

TEST_F(InternalCallbacksAdapterTest, TextAndToolCall) {
  auto callbacks = InternalCallbacksAdapter::Create(
      model_data_processor_.get(),
      std::make_unique<UserMessageCallbacks>(output_, done_, status_),
      processor_args_);

  callbacks->OnNext(CreateResponses("this "));
  callbacks->OnNext(CreateResponses("is "));
  callbacks->OnNext(CreateResponses("some "));
  callbacks->OnNext(CreateResponses("text\n"));
  callbacks->OnNext(CreateResponses("```tool_code\n"));
  callbacks->OnNext(CreateResponses("tool_name"));
  callbacks->OnNext(CreateResponses("(x=1)"));
  callbacks->OnNext(CreateResponses("\n```"));

  EXPECT_THAT(output_, ElementsAre(TextMessage("this "), TextMessage("is "),
                                   TextMessage("some "), TextMessage("text\n"),
                                   nlohmann::ordered_json::parse(R"json({
                            "role": "assistant",
                            "tool_calls": [
                              {
                                "type": "function",
                                "function": {
                                  "name": "tool_name",
                                  "arguments": {
                                    "x": 1
                                  }
                                }
                              }
                            ]
                          })json")));
}

TEST_F(InternalCallbacksAdapterTest, SplitCodeFenceStart) {
  auto callbacks = InternalCallbacksAdapter::Create(
      model_data_processor_.get(),
      std::make_unique<UserMessageCallbacks>(output_, done_, status_),
      processor_args_);

  callbacks->OnNext(CreateResponses("```tool_"));
  callbacks->OnNext(CreateResponses("code\n"));
  callbacks->OnNext(CreateResponses("tool_name"));
  callbacks->OnNext(CreateResponses("(x=1)"));
  callbacks->OnNext(CreateResponses("\n```"));

  EXPECT_THAT(output_, ElementsAre(nlohmann::ordered_json::parse(R"json({
                "role": "assistant",
                "tool_calls": [
                  {
                    "type": "function",
                    "function": {
                      "name": "tool_name",
                      "arguments": {
                        "x": 1
                      }
                    }
                  }
                ]
              })json")));
}

TEST_F(InternalCallbacksAdapterTest, TextBeforeSplitCodeFenceStart) {
  auto callbacks = InternalCallbacksAdapter::Create(
      model_data_processor_.get(),
      std::make_unique<UserMessageCallbacks>(output_, done_, status_),
      processor_args_);

  callbacks->OnNext(CreateResponses("text```tool_"));
  callbacks->OnNext(CreateResponses("code\n"));
  callbacks->OnNext(CreateResponses("tool_name"));
  callbacks->OnNext(CreateResponses("(x=1)"));
  callbacks->OnNext(CreateResponses("\n```"));

  EXPECT_THAT(output_, ElementsAre(TextMessage("text"),
                                   nlohmann::ordered_json::parse(R"json({
                "role": "assistant",
                "tool_calls": [
                  {
                    "type": "function",
                    "function": {
                      "name": "tool_name",
                      "arguments": {
                        "x": 1
                      }
                    }
                  }
                ]
              })json")));
}

TEST_F(InternalCallbacksAdapterTest, ToolCallAfterSplitCodeFenceStart) {
  auto callbacks = InternalCallbacksAdapter::Create(
      model_data_processor_.get(),
      std::make_unique<UserMessageCallbacks>(output_, done_, status_),
      processor_args_);

  callbacks->OnNext(CreateResponses("```"));
  callbacks->OnNext(CreateResponses("tool_code\ntool_name"));
  callbacks->OnNext(CreateResponses("(x=1)"));
  callbacks->OnNext(CreateResponses("\n```"));

  EXPECT_THAT(output_, ElementsAre(nlohmann::ordered_json::parse(R"json({
                "role": "assistant",
                "tool_calls": [
                  {
                    "type": "function",
                    "function": {
                      "name": "tool_name",
                      "arguments": {
                        "x": 1
                      }
                    }
                  }
                ]
              })json")));
}

TEST_F(InternalCallbacksAdapterTest, TextOnBothSidesOfCodeFenceStart) {
  auto callbacks = InternalCallbacksAdapter::Create(
      model_data_processor_.get(),
      std::make_unique<UserMessageCallbacks>(output_, done_, status_),
      processor_args_);

  callbacks->OnNext(CreateResponses("text```tool_code\ntool_name"));
  callbacks->OnNext(CreateResponses("(x=1)"));
  callbacks->OnNext(CreateResponses("\n```"));

  EXPECT_THAT(output_, ElementsAre(TextMessage("text"),
                                   nlohmann::ordered_json::parse(R"json({
                "role": "assistant",
                "tool_calls": [
                  {
                    "type": "function",
                    "function": {
                      "name": "tool_name",
                      "arguments": {
                        "x": 1
                      }
                    }
                  }
                ]
              })json")));
}

TEST_F(InternalCallbacksAdapterTest, SplitCodeFenceEnd) {
  auto callbacks = InternalCallbacksAdapter::Create(
      model_data_processor_.get(),
      std::make_unique<UserMessageCallbacks>(output_, done_, status_),
      processor_args_);

  callbacks->OnNext(CreateResponses("```tool_code\n"));
  callbacks->OnNext(CreateResponses("tool_name(x=1)"));
  callbacks->OnNext(CreateResponses("\n`"));
  callbacks->OnNext(CreateResponses("``"));

  EXPECT_THAT(output_, ElementsAre(nlohmann::ordered_json::parse(R"json({
                "role": "assistant",
                "tool_calls": [
                  {
                    "type": "function",
                    "function": {
                      "name": "tool_name",
                      "arguments": {
                        "x": 1
                      }
                    }
                  }
                ]
              })json")));
}

TEST_F(InternalCallbacksAdapterTest, TextBeforeSplitCodeFenceEnd) {
  auto callbacks = InternalCallbacksAdapter::Create(
      model_data_processor_.get(),
      std::make_unique<UserMessageCallbacks>(output_, done_, status_),
      processor_args_);

  callbacks->OnNext(CreateResponses("```tool_code\n"));
  callbacks->OnNext(CreateResponses("tool_name(x="));
  callbacks->OnNext(CreateResponses("1)\n``"));
  callbacks->OnNext(CreateResponses("`"));

  EXPECT_THAT(output_, ElementsAre(nlohmann::ordered_json::parse(R"json({
                "role": "assistant",
                "tool_calls": [
                  {
                    "type": "function",
                    "function": {
                      "name": "tool_name",
                      "arguments": {
                        "x": 1
                      }
                    }
                  }
                ]
              })json")));
}

TEST_F(InternalCallbacksAdapterTest, TextAfterSplitCodeFenceEnd) {
  auto callbacks = InternalCallbacksAdapter::Create(
      model_data_processor_.get(),
      std::make_unique<UserMessageCallbacks>(output_, done_, status_),
      processor_args_);

  callbacks->OnNext(CreateResponses("```tool_code\n"));
  callbacks->OnNext(CreateResponses("tool_name(x=1)"));
  callbacks->OnNext(CreateResponses("\n`"));
  callbacks->OnNext(CreateResponses("``text"));

  EXPECT_THAT(output_, ElementsAre(nlohmann::ordered_json::parse(R"json({
                            "role": "assistant",
                            "tool_calls": [
                              {
                                "type": "function",
                                "function": {
                                  "name": "tool_name",
                                  "arguments": {
                                    "x": 1
                                  }
                                }
                              }
                            ]
                          })json"),
                                   TextMessage("text")));
}

TEST_F(InternalCallbacksAdapterTest, OnNextTextOnBothSidesOfSplitCodeFenceEnd) {
  auto callbacks = InternalCallbacksAdapter::Create(
      model_data_processor_.get(),
      std::make_unique<UserMessageCallbacks>(output_, done_, status_),
      processor_args_);

  callbacks->OnNext(CreateResponses("```tool_code\n"));
  callbacks->OnNext(CreateResponses("tool_name(x="));
  callbacks->OnNext(CreateResponses("1)\n`"));
  callbacks->OnNext(CreateResponses("``text"));

  EXPECT_THAT(output_, ElementsAre(nlohmann::ordered_json::parse(R"json({
                            "role": "assistant",
                            "tool_calls": [
                              {
                                "type": "function",
                                "function": {
                                  "name": "tool_name",
                                  "arguments": {
                                    "x": 1
                                  }
                                }
                              }
                            ]
                          })json"),
                                   TextMessage("text")));
}

TEST_F(InternalCallbacksAdapterTest, ParallelToolCalls) {
  auto callbacks = InternalCallbacksAdapter::Create(
      model_data_processor_.get(),
      std::make_unique<UserMessageCallbacks>(output_, done_, status_),
      processor_args_);

  callbacks->OnNext(CreateResponses("```tool_code\n"));
  callbacks->OnNext(CreateResponses("tool_a(x=1)\n"));
  callbacks->OnNext(CreateResponses("tool_b(y='z')"));
  callbacks->OnNext(CreateResponses("\n```"));

  EXPECT_THAT(output_, ElementsAre(nlohmann::ordered_json::parse(R"json(
                {
                  "role": "assistant",
                  "tool_calls": [
                    {
                      "type": "function",
                      "function": {
                        "name": "tool_a",
                        "arguments": {
                          "x": 1
                        }
                      }
                    },
                    {
                      "type": "function",
                      "function": {
                        "name": "tool_b",
                        "arguments": {
                          "y": "z"
                        }
                      }
                    }
                  ]
                }
                )json")));
}

TEST_F(InternalCallbacksAdapterTest, TwoConsecutiveToolCodeBlocks) {
  auto callbacks = InternalCallbacksAdapter::Create(
      model_data_processor_.get(),
      std::make_unique<UserMessageCallbacks>(output_, done_, status_),
      processor_args_);

  callbacks->OnNext(CreateResponses("```tool_code\n"));
  callbacks->OnNext(CreateResponses("tool_a(x=1)\n"));
  callbacks->OnNext(CreateResponses("``````tool_code\n"));
  callbacks->OnNext(CreateResponses("tool_b(y='z')\n"));
  callbacks->OnNext(CreateResponses("```"));

  EXPECT_THAT(output_, ElementsAre(nlohmann::ordered_json::parse(R"json({
                            "role": "assistant",
                            "tool_calls": [
                              {
                                "type": "function",
                                "function": {
                                  "name": "tool_a",
                                  "arguments": {
                                    "x": 1
                                  }
                                }
                              }
                            ]
                          })json"),
                                   nlohmann::ordered_json::parse(R"json({
                            "role": "assistant",
                            "tool_calls": [
                              {
                                "type": "function",
                                "function": {
                                  "name": "tool_b",
                                  "arguments": {
                                    "y": "z"
                                  }
                                }
                              }
                            ]
                          })json")));
}

TEST_F(InternalCallbacksAdapterTest, IncompleteToolCodeBlock) {
  auto callbacks = InternalCallbacksAdapter::Create(
      model_data_processor_.get(),
      std::make_unique<UserMessageCallbacks>(output_, done_, status_),
      processor_args_);

  callbacks->OnNext(CreateResponses("```tool_code\n"));
  callbacks->OnNext(CreateResponses("tool_name(x=1)"));
  callbacks->OnDone();

  // The incomplete tool code block is sent to the callbacks as a text message.
  EXPECT_THAT(output_,
              ElementsAre(TextMessage("```tool_code\ntool_name(x=1)")));
}

TEST_F(InternalCallbacksAdapterTest, WrongCodeFenceStart) {
  auto callbacks = InternalCallbacksAdapter::Create(
      model_data_processor_.get(),
      std::make_unique<UserMessageCallbacks>(output_, done_, status_),
      processor_args_);

  callbacks->OnNext(CreateResponses("```tool\n"));
  callbacks->OnNext(CreateResponses("tool_name(x=1)"));
  callbacks->OnNext(CreateResponses("\n```"));
  callbacks->OnDone();

  EXPECT_THAT(output_, ElementsAre(TextMessage("```tool\n"),
                                   TextMessage("tool_name(x=1)"),
                                   TextMessage("\n"), TextMessage("```")));
}

TEST_F(InternalCallbacksAdapterTest, WrongCodeFenceEnd) {
  auto callbacks = InternalCallbacksAdapter::Create(
      model_data_processor_.get(),
      std::make_unique<UserMessageCallbacks>(output_, done_, status_),
      processor_args_);

  callbacks->OnNext(CreateResponses("```tool_code\n"));
  callbacks->OnNext(CreateResponses("tool_name(x=1)"));
  callbacks->OnNext(CreateResponses("\n``x"));
  callbacks->OnDone();

  EXPECT_THAT(output_,
              ElementsAre(TextMessage("```tool_code\ntool_name(x=1)\n``x")));
}

TEST_F(InternalCallbacksAdapterTest, InvalidFunctionCall) {
  auto callbacks = InternalCallbacksAdapter::Create(
      model_data_processor_.get(),
      std::make_unique<UserMessageCallbacks>(output_, done_, status_),
      processor_args_);

  callbacks->OnNext(CreateResponses("```tool_code\n"));
  callbacks->OnNext(CreateResponses("not a function call"));
  callbacks->OnNext(CreateResponses("\n```"));

  EXPECT_TRUE(done_);
  EXPECT_THAT(status_, StatusIs(absl::StatusCode::kInvalidArgument));
}

}  // namespace
}  // namespace litert::lm
