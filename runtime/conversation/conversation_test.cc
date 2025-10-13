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

#include <memory>
#include <string>
#include <utility>
#include <variant>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/synchronization/notification.h"  // from @com_google_absl
#include "absl/time/clock.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "nlohmann/json.hpp"  // from @nlohmann_json
#include "runtime/conversation/io_types.h"
#include "runtime/engine/engine.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/engine/io_types.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/util/test_utils.h"  // NOLINT

namespace litert::lm {
namespace {

absl::string_view kTestLlmPath =
    "litert_lm/runtime/testdata/test_lm.litertlm";

std::string GetTestdataPath(absl::string_view file_path) {
  return absl::StrCat(::testing::SrcDir(), "/", file_path);
}

class TestMessageCallbacks : public MessageCallbacks {
 public:
  explicit TestMessageCallbacks(const Message& expected_message)
      : expected_message_(expected_message) {}

  void OnError(const absl::Status& status) override {
    FAIL() << "OnError: " << status.message();
  }

  void OnMessage(const Message& message) override {
    const JsonMessage& json_message = std::get<JsonMessage>(message);
    JsonMessage& expected_json_message =
        std::get<JsonMessage>(expected_message_);
    // Compare the message text content by prefix, and update the expected
    // message to the remaining text for the next callback.
    ASSERT_TRUE(expected_json_message["content"][0]["text"].is_string());
    ASSERT_TRUE(json_message["content"][0]["text"].is_string());
    std::string expected_string = expected_json_message["content"][0]["text"];
    std::string actual_string = json_message["content"][0]["text"];
    EXPECT_TRUE(absl::StartsWith(expected_string, actual_string))
        << "Expected: " << expected_string << "\nActual: " << actual_string;
    expected_json_message["content"][0]["text"] =
        expected_string.substr(actual_string.size());
  }

  void OnComplete() override {
    JsonMessage& expected_json_message =
        std::get<JsonMessage>(expected_message_);
    ASSERT_TRUE(expected_json_message["content"][0]["text"].is_string());
    std::string expected_string = expected_json_message["content"][0]["text"];
    // The expected string should be empty after the last callback.
    EXPECT_TRUE(expected_string.empty());
    done_.Notify();
  }

 private:
  Message expected_message_;
  absl::Notification done_;
};

TEST(ConversationTest, SendMessage) {
  ASSERT_OK_AND_ASSIGN(auto model_assets,
                       ModelAssets::Create(GetTestdataPath(kTestLlmPath)));
  ASSERT_OK_AND_ASSIGN(auto engine_settings, EngineSettings::CreateDefault(
                                                 model_assets, Backend::CPU));
  engine_settings.GetMutableMainExecutorSettings().SetCacheDir(":nocache");
  engine_settings.GetMutableMainExecutorSettings().SetMaxNumTokens(10);
  ASSERT_OK_AND_ASSIGN(auto engine, Engine::CreateEngine(engine_settings));
  ASSERT_OK_AND_ASSIGN(auto session,
                       engine->CreateSession(SessionConfig::CreateDefault()));

  ASSERT_OK_AND_ASSIGN(auto conversation,
                       Conversation::Create(std::move(session)));
  EXPECT_THAT(conversation->GetHistory(), testing::IsEmpty());
  JsonMessage user_message = {{"role", "user"}, {"content", "Hello world!"}};
  ASSERT_OK_AND_ASSIGN(const Message message,
                       conversation->SendMessage(user_message));
  // The expected message is just some gibberish text, because the test LLM has
  // random weights.
  JsonMessage expected_message = {
      {"role", "assistant"},
      {"content",
       {{{"type", "text"}, {"text", "TarefaByte دارایेत्र investigaciónప్రదేశ"}}}}};
  const JsonMessage& json_message = std::get<JsonMessage>(message);
  EXPECT_EQ(json_message, expected_message);
  EXPECT_THAT(conversation->GetHistory(),
              testing::ElementsAre(user_message, expected_message));
}

TEST(ConversationTest, SendMessageStream) {
  ASSERT_OK_AND_ASSIGN(auto model_assets,
                       ModelAssets::Create(GetTestdataPath(kTestLlmPath)));
  ASSERT_OK_AND_ASSIGN(auto engine_settings, EngineSettings::CreateDefault(
                                                 model_assets, Backend::CPU));
  engine_settings.GetMutableMainExecutorSettings().SetCacheDir(":nocache");
  engine_settings.GetMutableMainExecutorSettings().SetMaxNumTokens(10);
  ASSERT_OK_AND_ASSIGN(auto engine, Engine::CreateEngine(engine_settings));
  ASSERT_OK_AND_ASSIGN(auto session,
                       engine->CreateSession(SessionConfig::CreateDefault()));
  ASSERT_OK_AND_ASSIGN(auto conversation,
                       Conversation::Create(std::move(session)));

  JsonMessage user_message = {{"role", "user"}, {"content", "Hello world!"}};
  // The expected message is just some gibberish text, because the test LLM has
  // random weights.
  JsonMessage expected_message = {
      {"role", "assistant"},
      {"content",
       {{{"type", "text"}, {"text", "TarefaByte دارایेत्र investigaciónప్రదేశ"}}}}};

  EXPECT_OK(conversation->SendMessageStream(
      user_message, std::make_unique<TestMessageCallbacks>(expected_message)));
  // Wait for the async message to be processed.
  EXPECT_OK(engine->WaitUntilDone(absl::Seconds(100)));
  EXPECT_THAT(conversation->GetHistory(),
              testing::ElementsAre(user_message, expected_message));
}

TEST(ConversationTest, SendMessageWithPreface) {
  ASSERT_OK_AND_ASSIGN(auto model_assets,
                       ModelAssets::Create(GetTestdataPath(kTestLlmPath)));
  ASSERT_OK_AND_ASSIGN(auto engine_settings, EngineSettings::CreateDefault(
                                                 model_assets, Backend::CPU));
  engine_settings.GetMutableMainExecutorSettings().SetCacheDir(":nocache");
  engine_settings.GetMutableMainExecutorSettings().SetMaxNumTokens(15);
  ASSERT_OK_AND_ASSIGN(auto engine, Engine::CreateEngine(engine_settings));
  ASSERT_OK_AND_ASSIGN(auto session,
                       engine->CreateSession(SessionConfig::CreateDefault()));
  Preface preface =
      JsonPreface{.messages = {{{"role", "system"},
                                {"content", "You are a helpful assistant."}}}};
  ASSERT_OK_AND_ASSIGN(auto conversation,
                       Conversation::Create(std::move(session), preface));
  ASSERT_OK_AND_ASSIGN(const Message message,
                       conversation->SendMessage(JsonMessage{
                           {"role", "user"}, {"content", "Hello world!"}}));
  // The expected message is just some gibberish text, because the test LLM has
  // random weights.
  JsonMessage expected_message = {
      {"role", "assistant"},
      {"content",
       {{{"type", "text"},
         {"text", " noses</caption> গ্রাহ<unused5297> omp"}}}}};
  const JsonMessage& json_message = std::get<JsonMessage>(message);
  EXPECT_EQ(json_message, expected_message);
}

TEST(ConversationTest, GetBenchmarkInfo) {
  ASSERT_OK_AND_ASSIGN(auto model_assets,
                       ModelAssets::Create(GetTestdataPath(kTestLlmPath)));
  ASSERT_OK_AND_ASSIGN(auto engine_settings, EngineSettings::CreateDefault(
                                                 model_assets, Backend::CPU));
  engine_settings.GetMutableMainExecutorSettings().SetCacheDir(":nocache");
  engine_settings.GetMutableMainExecutorSettings().SetMaxNumTokens(15);
  proto::BenchmarkParams benchmark_params;
  engine_settings.GetMutableBenchmarkParams() = benchmark_params;
  ASSERT_OK_AND_ASSIGN(auto engine, Engine::CreateEngine(engine_settings));
  ASSERT_OK_AND_ASSIGN(auto session,
                       engine->CreateSession(SessionConfig::CreateDefault()));
  Preface preface =
      JsonPreface{.messages = {{{"role", "system"},
                                {"content", "You are a helpful assistant."}}}};
  ASSERT_OK_AND_ASSIGN(auto conversation,
                       Conversation::Create(std::move(session), preface));
  ASSERT_OK_AND_ASSIGN(const Message message_1,
                       conversation->SendMessage(JsonMessage{
                           {"role", "user"}, {"content", "Hello world!"}}));
  ASSERT_OK_AND_ASSIGN(const BenchmarkInfo benchmark_info_1,
                       conversation->GetBenchmarkInfo());
  EXPECT_EQ(benchmark_info_1.GetTotalPrefillTurns(), 1);

  ASSERT_OK_AND_ASSIGN(const Message message_2,
                       conversation->SendMessage(JsonMessage{
                           {"role", "user"}, {"content", "Hello world!"}}));
  ASSERT_OK_AND_ASSIGN(const BenchmarkInfo benchmark_info_2,
                       conversation->GetBenchmarkInfo());
  EXPECT_EQ(benchmark_info_2.GetTotalPrefillTurns(), 2);
}

class CancelledMessageCallbacks : public MessageCallbacks {
 public:
  explicit CancelledMessageCallbacks(absl::Status& status,
                                     absl::Notification& done)
      : status_(status), done_(done) {}

  void OnError(const absl::Status& status) override {
    status_ = status;
    done_.Notify();
  }
  void OnMessage(const Message& message) override {
    // Wait for a short time to slow down the decoding process, so that the
    // cancellation can be triggered in the middle of decoding.
    absl::SleepFor(absl::Milliseconds(100));
  }

  void OnComplete() override {
    status_ = absl::OkStatus();
    done_.Notify();
  }

 private:
  absl::Status& status_;
  absl::Notification& done_;
};

class ConversationCancellationTest : public testing::TestWithParam<bool> {
 protected:
  bool use_benchmark_info_ = GetParam();
};

TEST_P(ConversationCancellationTest, CancelProcessWithBenchmarkInfo) {
  bool use_benchmark_info = use_benchmark_info_;
  ASSERT_OK_AND_ASSIGN(auto model_assets,
                       ModelAssets::Create(GetTestdataPath(kTestLlmPath)));
  ASSERT_OK_AND_ASSIGN(auto engine_settings, EngineSettings::CreateDefault(
                                                 model_assets, Backend::CPU));
  engine_settings.GetMutableMainExecutorSettings().SetCacheDir(":nocache");
  // Set a large max num tokens to ensure the decoding is not finished before
  // cancellation.
  engine_settings.GetMutableMainExecutorSettings().SetMaxNumTokens(20);
  if (use_benchmark_info) {
    proto::BenchmarkParams benchmark_params;
    engine_settings.GetMutableBenchmarkParams() = benchmark_params;
  }
  ASSERT_OK_AND_ASSIGN(auto engine, Engine::CreateEngine(engine_settings));
  ASSERT_OK_AND_ASSIGN(auto session,
                       engine->CreateSession(SessionConfig::CreateDefault()));
  ASSERT_OK_AND_ASSIGN(auto conversation,
                       Conversation::Create(std::move(session)));

  absl::Status status;
  absl::Notification done_1;
  conversation
      ->SendMessageStream(
          JsonMessage{{"role", "user"}, {"content", "Hello world!"}},
          std::make_unique<CancelledMessageCallbacks>(status, done_1))
      .IgnoreError();
  // Wait for a short time to ensure the decoding has started.
  absl::SleepFor(absl::Milliseconds(100));
  conversation->CancelProcess();
  // Wait for the callbacks to be done.
  done_1.WaitForNotification();
  EXPECT_THAT(status, testing::status::StatusIs(absl::StatusCode::kCancelled));

  // The history should be empty after cancellation.
  EXPECT_THAT(conversation->GetHistory().size(), 0);

  // Re-send the message after cancellation, and it should succeed.
  status = absl::OkStatus();
  absl::Notification done_2;
  conversation
      ->SendMessageStream(
          JsonMessage{{"role", "user"}, {"content", "Hello world!"}},
          std::make_unique<CancelledMessageCallbacks>(status, done_2))
      .IgnoreError();
  EXPECT_OK(status);
  // Wait for the callbacks to be done.
  done_2.WaitForNotification();
  // Without cancellation, the history should have two messages, user and
  // assistant.
  auto history = conversation->GetHistory();
  ASSERT_EQ(history.size(), 2);
  EXPECT_THAT(history[0], testing::VariantWith<JsonMessage>(JsonMessage{
                              {"role", "user"}, {"content", "Hello world!"}}));
  // TODO(b/450903294) - Because the cancellation is not fully rollbacked, the
  // assistant message content depends on at which step the cancellation is
  // triggered, and that is non-deterministic. Here we only check the role is
  // assistant.
  EXPECT_THAT(std::holds_alternative<JsonMessage>(history[1]),
              testing::IsTrue());
  EXPECT_EQ(std::get<JsonMessage>(history[1])["role"], "assistant");

  conversation->CancelProcess();
  // No op after cancellation again.
  EXPECT_THAT(conversation->GetHistory().size(), 2);
}

INSTANTIATE_TEST_SUITE_P(ConversationCancellationTest,
                         ConversationCancellationTest, testing::Bool(),
                         testing::PrintToStringParamName());

}  // namespace
}  // namespace litert::lm
