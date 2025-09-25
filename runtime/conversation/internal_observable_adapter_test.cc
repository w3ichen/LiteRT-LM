#include "runtime/conversation/internal_observable_adapter.h"

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

nlohmann::ordered_json ToolCallMessage(nlohmann::ordered_json tool_call) {
  nlohmann::ordered_json message;
  message["role"] = "assistant";
  message["content"] = {{{"type", "tool_call"}, {"tool_call", tool_call}}};
  return message;
}

class UserMessageObservable : public MessageObservable {
 public:
  void OnMessage(const Message& message) override {
    output_.push_back(std::get<nlohmann::ordered_json>(message));
  }
  void OnComplete() override { done_ = true; }
  void OnError(const absl::Status& status) override {
    done_ = true;
    status_ = status;
  }

  const std::vector<nlohmann::ordered_json>& output() const { return output_; }
  bool done() const { return done_; }
  absl::Status status() const { return status_; }

 private:
  std::vector<nlohmann::ordered_json> output_;
  bool done_ = false;
  absl::Status status_ = absl::OkStatus();
};

class InternalObservableAdapterTest : public testing::Test {
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
    user_observer_ = std::make_unique<UserMessageObservable>();
    processor_args_ = DataProcessorArguments();
  }

  std::unique_ptr<Gemma3DataProcessor> model_data_processor_;
  std::unique_ptr<UserMessageObservable> user_observer_;
  DataProcessorArguments processor_args_;
};

TEST_F(InternalObservableAdapterTest, OnDone) {
  auto observer = InternalObservableAdapter::Create(
      model_data_processor_.get(), user_observer_.get(), processor_args_);

  observer->OnDone();

  EXPECT_THAT(user_observer_->output(), IsEmpty());
  EXPECT_TRUE(user_observer_->done());
  EXPECT_OK(user_observer_->status());
}

TEST_F(InternalObservableAdapterTest, OnError) {
  auto observer = InternalObservableAdapter::Create(
      model_data_processor_.get(), user_observer_.get(), processor_args_);

  observer->OnError(absl::InternalError("error"));

  EXPECT_THAT(user_observer_->output(), IsEmpty());
  EXPECT_TRUE(user_observer_->done());
  EXPECT_THAT(user_observer_->status(),
              StatusIs(absl::StatusCode::kInternal, "error"));
}

TEST_F(InternalObservableAdapterTest, Text) {
  auto observer = InternalObservableAdapter::Create(
      model_data_processor_.get(), user_observer_.get(), processor_args_);

  observer->OnNext(CreateResponses("this "));
  observer->OnNext(CreateResponses("is "));
  observer->OnNext(CreateResponses("some "));
  observer->OnNext(CreateResponses("text"));

  EXPECT_THAT(user_observer_->output(),
              ElementsAre(TextMessage("this "), TextMessage("is "),
                          TextMessage("some "), TextMessage("text")));
}

TEST_F(InternalObservableAdapterTest, ToolCall) {
  auto observer = InternalObservableAdapter::Create(
      model_data_processor_.get(), user_observer_.get(), processor_args_);

  observer->OnNext(CreateResponses("```tool_code\n"));
  observer->OnNext(CreateResponses("tool_name"));
  observer->OnNext(CreateResponses("(x=1)"));
  observer->OnNext(CreateResponses("\n```"));

  EXPECT_THAT(user_observer_->output(),
              ElementsAre(ToolCallMessage(nlohmann::ordered_json::parse(R"json(
                {
                  "name": "tool_name",
                  "args": {
                    "x": 1
                  }
                }
              )json"))));
}

TEST_F(InternalObservableAdapterTest, TextAndToolCall) {
  auto observer = InternalObservableAdapter::Create(
      model_data_processor_.get(), user_observer_.get(), processor_args_);

  observer->OnNext(CreateResponses("this "));
  observer->OnNext(CreateResponses("is "));
  observer->OnNext(CreateResponses("some "));
  observer->OnNext(CreateResponses("text\n"));
  observer->OnNext(CreateResponses("```tool_code\n"));
  observer->OnNext(CreateResponses("tool_name"));
  observer->OnNext(CreateResponses("(x=1)"));
  observer->OnNext(CreateResponses("\n```"));

  EXPECT_THAT(user_observer_->output(),
              ElementsAre(TextMessage("this "), TextMessage("is "),
                          TextMessage("some "), TextMessage("text\n"),
                          ToolCallMessage(nlohmann::ordered_json::parse(R"json(
                {
                  "name": "tool_name",
                  "args": {
                    "x": 1
                  }
                }
              )json"))));
}

TEST_F(InternalObservableAdapterTest, SplitCodeFenceStart) {
  auto observer = InternalObservableAdapter::Create(
      model_data_processor_.get(), user_observer_.get(), processor_args_);

  observer->OnNext(CreateResponses("```tool_"));
  observer->OnNext(CreateResponses("code\n"));
  observer->OnNext(CreateResponses("tool_name"));
  observer->OnNext(CreateResponses("(x=1)"));
  observer->OnNext(CreateResponses("\n```"));

  EXPECT_THAT(user_observer_->output(),
              ElementsAre(ToolCallMessage(nlohmann::ordered_json::parse(R"json(
                {
                  "name": "tool_name",
                  "args": {
                    "x": 1
                  }
                }
              )json"))));
}

TEST_F(InternalObservableAdapterTest, TextBeforeSplitCodeFenceStart) {
  auto observer = InternalObservableAdapter::Create(
      model_data_processor_.get(), user_observer_.get(), processor_args_);

  observer->OnNext(CreateResponses("text```tool_"));
  observer->OnNext(CreateResponses("code\n"));
  observer->OnNext(CreateResponses("tool_name"));
  observer->OnNext(CreateResponses("(x=1)"));
  observer->OnNext(CreateResponses("\n```"));

  EXPECT_THAT(user_observer_->output(),
              ElementsAre(TextMessage("text"),
                          ToolCallMessage(nlohmann::ordered_json::parse(R"json(
                {
                  "name": "tool_name",
                  "args": {
                    "x": 1
                  }
                }
              )json"))));
}

TEST_F(InternalObservableAdapterTest, ToolCallAfterSplitCodeFenceStart) {
  auto observer = InternalObservableAdapter::Create(
      model_data_processor_.get(), user_observer_.get(), processor_args_);

  observer->OnNext(CreateResponses("```"));
  observer->OnNext(CreateResponses("tool_code\ntool_name"));
  observer->OnNext(CreateResponses("(x=1)"));
  observer->OnNext(CreateResponses("\n```"));

  EXPECT_THAT(user_observer_->output(),
              ElementsAre(ToolCallMessage(nlohmann::ordered_json::parse(R"json(
                {
                  "name": "tool_name",
                  "args": {
                    "x": 1
                  }
                }
              )json"))));
}

TEST_F(InternalObservableAdapterTest, TextOnBothSidesOfCodeFenceStart) {
  auto observer = InternalObservableAdapter::Create(
      model_data_processor_.get(), user_observer_.get(), processor_args_);

  observer->OnNext(CreateResponses("text```tool_code\ntool_name"));
  observer->OnNext(CreateResponses("(x=1)"));
  observer->OnNext(CreateResponses("\n```"));

  EXPECT_THAT(user_observer_->output(),
              ElementsAre(TextMessage("text"),
                          ToolCallMessage(nlohmann::ordered_json::parse(R"json(
                {
                  "name": "tool_name",
                  "args": {
                    "x": 1
                  }
                }
              )json"))));
}

TEST_F(InternalObservableAdapterTest, SplitCodeFenceEnd) {
  auto observer = InternalObservableAdapter::Create(
      model_data_processor_.get(), user_observer_.get(), processor_args_);

  observer->OnNext(CreateResponses("```tool_code\n"));
  observer->OnNext(CreateResponses("tool_name(x=1)"));
  observer->OnNext(CreateResponses("\n`"));
  observer->OnNext(CreateResponses("``"));

  EXPECT_THAT(user_observer_->output(),
              ElementsAre(ToolCallMessage(nlohmann::ordered_json::parse(R"json(
                {
                  "name": "tool_name",
                  "args": {
                    "x": 1
                  }
                }
              )json"))));
}

TEST_F(InternalObservableAdapterTest, TextBeforeSplitCodeFenceEnd) {
  auto observer = InternalObservableAdapter::Create(
      model_data_processor_.get(), user_observer_.get(), processor_args_);

  observer->OnNext(CreateResponses("```tool_code\n"));
  observer->OnNext(CreateResponses("tool_name(x="));
  observer->OnNext(CreateResponses("1)\n``"));
  observer->OnNext(CreateResponses("`"));

  EXPECT_THAT(user_observer_->output(),
              ElementsAre(ToolCallMessage(nlohmann::ordered_json::parse(R"json(
                {
                  "name": "tool_name",
                  "args": {
                    "x": 1
                  }
                }
              )json"))));
}

TEST_F(InternalObservableAdapterTest, TextAfterSplitCodeFenceEnd) {
  auto observer = InternalObservableAdapter::Create(
      model_data_processor_.get(), user_observer_.get(), processor_args_);

  observer->OnNext(CreateResponses("```tool_code\n"));
  observer->OnNext(CreateResponses("tool_name(x=1)"));
  observer->OnNext(CreateResponses("\n`"));
  observer->OnNext(CreateResponses("``text"));

  EXPECT_THAT(user_observer_->output(),
              ElementsAre(ToolCallMessage(nlohmann::ordered_json::parse(R"json(
                {
                  "name": "tool_name",
                  "args": {
                    "x": 1
                  }
                }
              )json")),
                          TextMessage("text")));
}

TEST_F(InternalObservableAdapterTest,
       OnNextTextOnBothSidesOfSplitCodeFenceEnd) {
  auto observer = InternalObservableAdapter::Create(
      model_data_processor_.get(), user_observer_.get(), processor_args_);

  observer->OnNext(CreateResponses("```tool_code\n"));
  observer->OnNext(CreateResponses("tool_name(x="));
  observer->OnNext(CreateResponses("1)\n`"));
  observer->OnNext(CreateResponses("``text"));

  EXPECT_THAT(user_observer_->output(),
              ElementsAre(ToolCallMessage(nlohmann::ordered_json::parse(R"json(
                {
                  "name": "tool_name",
                  "args": {
                    "x": 1
                  }
                }
              )json")),
                          TextMessage("text")));
}

TEST_F(InternalObservableAdapterTest, ParallelToolCalls) {
  auto observer = InternalObservableAdapter::Create(
      model_data_processor_.get(), user_observer_.get(), processor_args_);

  observer->OnNext(CreateResponses("```tool_code\n"));
  observer->OnNext(CreateResponses("tool_a(x=1)\n"));
  observer->OnNext(CreateResponses("tool_b(y='z')"));
  observer->OnNext(CreateResponses("\n```"));

  EXPECT_THAT(user_observer_->output(),
              ElementsAre(nlohmann::ordered_json::parse(R"json(
                {
                  "role": "assistant",
                  "content": [
                    {
                      "type": "tool_call",
                      "tool_call": {
                        "name": "tool_a",
                        "args": {
                          "x": 1
                        }
                      }
                    },
                    {
                      "type": "tool_call",
                      "tool_call": {
                        "name": "tool_b",
                        "args": {
                          "y": "z"
                        }
                      }
                    }
                  ]
                }
                )json")));
}

TEST_F(InternalObservableAdapterTest, TwoConsecutiveToolCodeBlocks) {
  auto observer = InternalObservableAdapter::Create(
      model_data_processor_.get(), user_observer_.get(), processor_args_);

  observer->OnNext(CreateResponses("```tool_code\n"));
  observer->OnNext(CreateResponses("tool_a(x=1)\n"));
  observer->OnNext(CreateResponses("``````tool_code\n"));
  observer->OnNext(CreateResponses("tool_b(y='z')\n"));
  observer->OnNext(CreateResponses("```"));

  EXPECT_THAT(user_observer_->output(),
              ElementsAre(ToolCallMessage(nlohmann::ordered_json::parse(R"json(
                    {
                      "name": "tool_a",
                      "args": {
                        "x": 1
                      }
                    }
                  )json")),
                          ToolCallMessage(nlohmann::ordered_json::parse(R"json(
                    {
                      "name": "tool_b",
                      "args": {
                        "y": "z"
                      }
                    }
                  )json"))));
}

TEST_F(InternalObservableAdapterTest, IncompleteToolCodeBlock) {
  auto observer = InternalObservableAdapter::Create(
      model_data_processor_.get(), user_observer_.get(), processor_args_);

  observer->OnNext(CreateResponses("```tool_code\n"));
  observer->OnNext(CreateResponses("tool_name(x=1)"));

  EXPECT_THAT(user_observer_->output(), IsEmpty());
  // TODO: Remainder is everything.
}

TEST_F(InternalObservableAdapterTest, WrongCodeFenceStart) {
  auto observer = InternalObservableAdapter::Create(
      model_data_processor_.get(), user_observer_.get(), processor_args_);

  observer->OnNext(CreateResponses("```tool\n"));
  observer->OnNext(CreateResponses("tool_name(x=1)"));
  observer->OnNext(CreateResponses("\n```"));
  observer->OnDone();

  EXPECT_THAT(
      user_observer_->output(),
      ElementsAre(TextMessage("```tool\n"), TextMessage("tool_name(x=1)"),
                  TextMessage("\n"), TextMessage("```")));
}

TEST_F(InternalObservableAdapterTest, WrongCodeFenceEnd) {
  auto observer = InternalObservableAdapter::Create(
      model_data_processor_.get(), user_observer_.get(), processor_args_);

  observer->OnNext(CreateResponses("```tool_code\n"));
  observer->OnNext(CreateResponses("tool_name(x=1)"));
  observer->OnNext(CreateResponses("\n``x"));
  observer->OnDone();

  EXPECT_THAT(user_observer_->output(),
              ElementsAre(TextMessage("```tool_code\ntool_name(x=1)\n``x")));
}

TEST_F(InternalObservableAdapterTest, InvalidFunctionCall) {
  auto observer = InternalObservableAdapter::Create(
      model_data_processor_.get(), user_observer_.get(), processor_args_);

  observer->OnNext(CreateResponses("```tool_code\n"));
  observer->OnNext(CreateResponses("not a function call"));
  observer->OnNext(CreateResponses("\n```"));

  EXPECT_TRUE(user_observer_->done());
  EXPECT_THAT(user_observer_->status(), StatusIs(absl::StatusCode::kInternal));
}

}  // namespace
}  // namespace litert::lm
