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

#include <filesystem>  // NOLINT: Required for path manipulation.
#include <fstream>
#include <ios>
#include <iterator>
#include <sstream>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "nlohmann/json.hpp"  // from @nlohmann_json
#include "litert/cc/litert_layout.h"  // from @litert
#include "runtime/components/preprocessor/audio_preprocessor.h"
#include "runtime/components/preprocessor/audio_preprocessor_miniaudio.h"
#include "runtime/components/preprocessor/image_preprocessor.h"
#include "runtime/components/preprocessor/stb_image_preprocessor.h"
#include "runtime/components/prompt_template.h"
#include "runtime/conversation/io_types.h"
#include "runtime/conversation/model_data_processor/gemma3_data_processor_config.h"
#include "runtime/engine/io_types.h"
#include "runtime/util/convert_tensor_buffer.h"
#include "runtime/util/test_utils.h"  // NOLINT

namespace litert::lm {
namespace {

using json = nlohmann::ordered_json;
using ::testing::ElementsAre;
using ::testing::status::IsOkAndHolds;

constexpr char kTestdataDir[] =
    "litert_lm/runtime/components/testdata/";

constexpr char kImageTestdataDir[] =
    "litert_lm/runtime/components/preprocessor/testdata/";

std::string GetTestdataPath(const std::string& file_name) {
  return (std::filesystem::path(::testing::SrcDir()) / kTestdataDir / file_name)
      .string();
}

std::string ReadFile(absl::string_view path) {
  std::ifstream ifstr(std::string(path), std::ios::binary);
  std::stringstream contents;
  contents << ifstr.rdbuf();
  return contents.str();
}

absl::StatusOr<std::string> GetContents(const std::string& path) {
  std::ifstream input_stream(path);
  if (!input_stream.is_open()) {
    return absl::InternalError(absl::StrCat("Could not open file: ", path));
  }

  std::string content;
  content.assign((std::istreambuf_iterator<char>(input_stream)),
                 (std::istreambuf_iterator<char>()));
  return std::move(content);
}

MATCHER_P(HasInputText, text_input, "") {
  if (!std::holds_alternative<InputText>(arg)) {
    return false;
  }
  auto text_bytes = std::get<InputText>(arg).GetRawTextString();
  if (!text_bytes.ok()) {
    return false;
  }
  return text_bytes.value() == text_input->GetRawTextString().value();
}

MATCHER_P(HasInputImage, image_input, "") {
  if (!std::holds_alternative<InputImage>(arg)) {
    return false;
  }
  if (std::get<InputImage>(arg).GetRawImageBytes().ok()) {
    auto image_bytes = std::get<InputImage>(arg).GetRawImageBytes();
    return image_bytes.value() == image_input->GetRawImageBytes().value();
  }
  if (std::get<InputImage>(arg).GetPreprocessedImageTensor().ok()) {
    auto buffer_span = ReferTensorBufferAsSpan<float>(
        *std::get<InputImage>(arg).GetPreprocessedImageTensor().value());
    if (!buffer_span.HasValue()) {
      return false;
    }
    auto expected_buffer_span = ReferTensorBufferAsSpan<float>(
        *image_input->GetPreprocessedImageTensor().value());
    if (!expected_buffer_span.HasValue()) {
      return false;
    }
    return *buffer_span == *expected_buffer_span;
  }
  return true;
}

MATCHER_P(HasInputAudio, audio_input, "") {
  if (!std::holds_alternative<InputAudio>(arg)) {
    return false;
  }
  if (std::get<InputAudio>(arg).GetRawAudioBytes().ok()) {
    auto audio_bytes = std::get<InputAudio>(arg).GetRawAudioBytes();
    return audio_bytes.value() == audio_input->GetRawAudioBytes().value();
  }
  if (std::get<InputAudio>(arg).GetPreprocessedAudioTensor().ok()) {
    auto buffer_span = ReferTensorBufferAsSpan<float>(
        *std::get<InputAudio>(arg).GetPreprocessedAudioTensor().value());
    if (!buffer_span.HasValue()) {
      return false;
    }
    auto expected_buffer_span = ReferTensorBufferAsSpan<float>(
        *audio_input->GetPreprocessedAudioTensor().value());
    if (!expected_buffer_span.HasValue()) {
      return false;
    }
    return *buffer_span == *expected_buffer_span;
  }
  return true;
}

TEST(Gemma3DataProcessorTest, ToInputDataVectorTextOnly) {
  ASSERT_OK_AND_ASSIGN(auto processor, Gemma3DataProcessor::Create());
  const std::string rendered_template_prompt =
      "<start_of_turn>user\ntest prompt\n<end_of_turn>";
  const nlohmann::ordered_json messages = {
      {"role", "user"},
      {"content", "test prompt"},
  };
  ASSERT_OK_AND_ASSIGN(
      const std::vector<InputData> input_data,
      processor->ToInputDataVector(rendered_template_prompt, messages, {}));

  InputText expected_text("<start_of_turn>user\ntest prompt\n<end_of_turn>");
  EXPECT_THAT(input_data, ElementsAre(HasInputText(&expected_text)));
}

TEST(Gemma3DataProcessorTest, ToInputDataVectorTextAndImage) {
  ASSERT_OK_AND_ASSIGN(auto processor, Gemma3DataProcessor::Create(
                                           /*Gemma3DataProcessorConfig=*/{
                                               .image_tensor_height = 224,
                                               .image_tensor_width = 128}));
  const std::string rendered_template_prompt =
      "<start_of_turn>user\nHere is an image of apples "
      "<start_of_image><end_of_turn>";

  std::string image_path = (std::filesystem::path(::testing::SrcDir()) /
                            kImageTestdataDir / "apple.png")
                               .string();
  const nlohmann::ordered_json message = {
      {"role", "user"},
      {"content",
       {{{"type", "text"}, {"text", "Here is an image of apples "}},
        {{"type", "image"}, {"path", image_path}}}}};
  ASSERT_OK_AND_ASSIGN(
      const std::vector<InputData> input_data,
      processor->ToInputDataVector(rendered_template_prompt,
                                   json::array({message}), {}));

  InputText expected_text1(
      "<start_of_turn>user\nHere is an image of apples <start_of_image>");
  StbImagePreprocessor image_preprocessor;
  ImagePreprocessParameter image_params;
  image_params.SetTargetDimensions(Dimensions({1, 224, 128, 3}));
  ASSERT_OK_AND_ASSIGN(InputImage expected_image,
                       image_preprocessor.Preprocess(
                           InputImage(ReadFile(image_path)), image_params));
  InputText expected_text2("<end_of_turn>");
  EXPECT_THAT(input_data, ElementsAre(HasInputText(&expected_text1),
                                      HasInputImage(&expected_image),
                                      HasInputText(&expected_text2)));
}

TEST(Gemma3DataProcessorTest, ToMessage) {
  ASSERT_OK_AND_ASSIGN(auto processor, Gemma3DataProcessor::Create());
  Responses responses(1);
  responses.GetMutableResponseTexts()[0] = "test response";
  ASSERT_OK_AND_ASSIGN(const Message message,
                       processor->ToMessage(responses, std::monostate{}));

  ASSERT_TRUE(std::holds_alternative<nlohmann::ordered_json>(message));
  const nlohmann::ordered_json& json_message =
      std::get<nlohmann::ordered_json>(message);
  EXPECT_EQ(
      json_message,
      json({{"role", "assistant"},
            {"content", {{{"type", "text"}, {"text", "test response"}}}}}));
}

TEST(Gemma3DataProcessorTest, ToMessageWithToolCall) {
  Gemma3DataProcessorConfig config;
  JsonPreface preface{.tools = nlohmann::ordered_json::parse(
                          R"json([{
                            "name": "tool_name",
                            "parameters": {
                              "properties": {
                                "x": {
                                  "type": "integer"
                                }
                              }
                            }
                          }])json")};

  ASSERT_OK_AND_ASSIGN(auto processor,
                       Gemma3DataProcessor::Create(config, preface));
  Responses responses(1);
  responses.GetMutableResponseTexts()[0] = R"(This is some text.
```tool_code
tool_name(x=1)
```)";

  ASSERT_OK_AND_ASSIGN(const Message message,
                       processor->ToMessage(responses, std::monostate{}));

  ASSERT_TRUE(std::holds_alternative<nlohmann::ordered_json>(message));
  const nlohmann::ordered_json& json_message =
      std::get<nlohmann::ordered_json>(message);
  EXPECT_EQ(json_message, nlohmann::ordered_json::parse(R"json({
    "role": "assistant",
    "content": [
      {
        "type": "text",
        "text": "This is some text.\n"
      }
    ],
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
  })json"));
}

TEST(Gemma3DataProcessorTest, PromptTemplateToInputDataVectorTextOnly) {
  const std::string test_file_path =
      GetTestdataPath("google-gemma-3-1b-it.jinja");
  ASSERT_OK_AND_ASSIGN(const std::string template_content,
                       GetContents(test_file_path));
  PromptTemplate prompt_template(template_content);

  const nlohmann::ordered_json messages = {
      {{"role", "system"}, {"content", "Hello world!"}},
      {{"role", "user"}, {"content", "How are you?"}},
      {{"role", "assistant"},
       {"content", "I am doing well, thanks for asking."}},
      {{"role", "user"}, {"content", "What is the capital of France?"}},
  };
  PromptTemplateInput template_input = {.messages = messages,
                                        .add_generation_prompt = true};

  ASSERT_OK_AND_ASSIGN(const std::string rendered_prompt,
                       prompt_template.Apply(template_input));

  ASSERT_OK_AND_ASSIGN(auto processor, Gemma3DataProcessor::Create());
  ASSERT_OK_AND_ASSIGN(
      const std::vector<InputData> input_data,
      processor->ToInputDataVector(rendered_prompt, messages, {}));
  InputText expected_text(R"""(<start_of_turn>user
Hello world!

How are you?<end_of_turn>
<start_of_turn>model
I am doing well, thanks for asking.<end_of_turn>
<start_of_turn>user
What is the capital of France?<end_of_turn>
<start_of_turn>model
)""");
  EXPECT_THAT(input_data, ElementsAre(HasInputText(&expected_text)));
}

TEST(Gemma3DataProcessorTest, PromptTemplateToInputDataVectorTextAndImage) {
  const std::string test_file_path =
      GetTestdataPath("google-gemma-3-1b-it.jinja");
  ASSERT_OK_AND_ASSIGN(const std::string template_content,
                       GetContents(test_file_path));
  PromptTemplate prompt_template(template_content);

  std::string image_path = (std::filesystem::path(::testing::SrcDir()) /
                            kImageTestdataDir / "apple.png")
                               .string();
  const nlohmann::ordered_json messages = {
      {{"role", "system"}, {"content", "Hello world!"}},
      {{"role", "user"},
       {"content",
        {{{"type", "text"}, {"text", "How are you?"}},
         {{"type", "image"}, {"path", image_path}}}}},
      {{"role", "assistant"},
       {"content", "I am doing well, thanks for asking."}},
      {{"role", "user"},
       {"content",
        {{{"type", "image"}, {"path", image_path}},
         {{"type", "text"}, {"text", "What is the capital of France?"}}}}}};
  PromptTemplateInput template_input = {.messages = messages,
                                        .add_generation_prompt = true};

  ASSERT_OK_AND_ASSIGN(const std::string rendered_prompt,
                       prompt_template.Apply(template_input));

  ASSERT_OK_AND_ASSIGN(auto processor, Gemma3DataProcessor::Create());
  ASSERT_OK_AND_ASSIGN(
      const std::vector<InputData> input_data,
      processor->ToInputDataVector(rendered_prompt, messages, {}));
  InputText expected_text1(R"""(<start_of_turn>user
Hello world!

How are you?<start_of_image>)""");
  StbImagePreprocessor image_preprocessor;
  ImagePreprocessParameter image_params;
  image_params.SetTargetDimensions(Dimensions({1, 768, 768, 3}));
  ASSERT_OK_AND_ASSIGN(InputImage expected_image,
                       image_preprocessor.Preprocess(
                           InputImage(ReadFile(image_path)), image_params));
  InputText expected_text2(R"""(<end_of_turn>
<start_of_turn>model
I am doing well, thanks for asking.<end_of_turn>
<start_of_turn>user
<start_of_image>)""");
  InputText expected_text3(R"""(What is the capital of France?<end_of_turn>
<start_of_turn>model
)""");
  EXPECT_THAT(
      input_data,
      ElementsAre(HasInputText(&expected_text1), HasInputImage(&expected_image),
                  HasInputText(&expected_text2), HasInputImage(&expected_image),
                  HasInputText(&expected_text3)));
}

TEST(Gemma3DataProcessorTest, FormatTools) {
  ASSERT_OK_AND_ASSIGN(auto processor, Gemma3DataProcessor::Create());
  nlohmann::ordered_json tools = nlohmann::ordered_json::parse(R"json([
    {
      "name": "get_weather",
      "description": "Gets weather information.",
      "parameters": {
        "properties": {
          "location": {
            "type": "string",
            "description": "Weather location."
          }
        },
        "required": ["location"]
      }
    },
    {
      "name": "get_stock_price",
      "description": "Gets stock price.",
      "parameters": {
        "properties": {
          "symbol": {
            "type": "string",
            "description": "Stock symbol."
          }
        },
        "required": ["symbol"]
      }
    }
  ])json");

  ASSERT_OK_AND_ASSIGN(const nlohmann::ordered_json formatted_tools,
                       processor->FormatTools(tools));

  nlohmann::ordered_json expected = nlohmann::ordered_json::array();
  expected.push_back(R"(def get_weather(
    location: str,
) -> dict:
  """Gets weather information.

  Args:
    location: Weather location.
  """
)");
  expected.push_back(R"(def get_stock_price(
    symbol: str,
) -> dict:
  """Gets stock price.

  Args:
    symbol: Stock symbol.
  """
)");

  EXPECT_EQ(formatted_tools, expected);
}

TEST(Gemma3DataProcessorTest, FormatToolsWithInvalidInput) {
  ASSERT_OK_AND_ASSIGN(auto processor, Gemma3DataProcessor::Create());
  // `tools` is not an array.
  nlohmann::ordered_json tools = nlohmann::ordered_json::parse(R"json({
    "name": "get_weather",
    "description": "Gets weather information.",
    "parameters": {
      "properties": {
        "location": {
          "type": "string",
          "description": "Weather location."
        }
      },
      "required": ["location"]
    }
  })json");

  EXPECT_THAT(processor->FormatTools(tools),
              testing::status::StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(Gemma3DataProcessorTest, MessageToTemplateInputWithStringContent) {
  ASSERT_OK_AND_ASSIGN(auto processor, Gemma3DataProcessor::Create());
  const nlohmann::ordered_json message = {
      {"role", "user"},
      {"content", "test prompt"},
  };

  // The template input is identical to the original message if the content is
  // a string.
  EXPECT_THAT(processor->MessageToTemplateInput(message),
              IsOkAndHolds(message));
}

TEST(Gemma3DataProcessorTest, MessageToTemplateInputWithTextContent) {
  ASSERT_OK_AND_ASSIGN(auto processor, Gemma3DataProcessor::Create());
  const nlohmann::ordered_json message = {
      {"role", "user"},
      {"content", {{{"type", "text"}, {"text", "test prompt"}}}},
  };

  // Text content items should be unchanged.
  EXPECT_THAT(processor->MessageToTemplateInput(message),
              IsOkAndHolds(message));
}

TEST(Gemma3DataProcessorTest, MessageToTemplateInputNoContent) {
  ASSERT_OK_AND_ASSIGN(auto processor, Gemma3DataProcessor::Create());
  const nlohmann::ordered_json message = {
      {"role", "user"},
  };

  // The template input should be unchanged if there is no content.
  EXPECT_THAT(processor->MessageToTemplateInput(message),
              IsOkAndHolds(message));
}

TEST(Gemma3DataProcessorTest, MessageToTemplateInputWithToolCalls) {
  ASSERT_OK_AND_ASSIGN(auto processor, Gemma3DataProcessor::Create());
  const nlohmann::ordered_json message = nlohmann::ordered_json::parse(R"json({
    "role": "assistant",
    "content": [
      {
        "type": "text",
        "text": "This is some text."
      }
    ],
    "tool_calls": [
      {
        "type": "function",
        "function": {
          "name": "tool1",
          "arguments": {
            "x": 1
          }
        }
      },
      {
        "type": "function",
        "function": {
          "name": "tool2",
          "arguments": {
            "y": "foo"
          }
        }
      }
    ]
  })json");

  EXPECT_THAT(processor->MessageToTemplateInput(message),
              IsOkAndHolds(nlohmann::ordered_json::parse(R"json({
  "role": "assistant",
  "content": [
    {
      "type": "text",
      "text": "This is some text."
    }
  ],
  "tool_calls": [
    {
      "type": "function",
      "function": {
        "name": "tool1",
        "arguments": {
          "x": "1"
        }
      }
    },
    {
      "type": "function",
      "function": {
        "name": "tool2",
        "arguments": {
          "y": "\"foo\""
        }
      }
    }
  ]
})json")));
}

TEST(Gemma3DataProcessorTest, MessageToTemplateInputWithToolResponse) {
  ASSERT_OK_AND_ASSIGN(auto processor, Gemma3DataProcessor::Create());
  const nlohmann::ordered_json message = nlohmann::ordered_json::parse(R"json({
    "role": "tool",
    "content": [
      {
        "type": "tool_response",
        "tool_response": {
          "key1": "value1",
          "key2": "value2"
        }
      }
    ]
  })json");

  // The template input should contain a tool_outputs item with the tool
  // response formatted as a Python dict.
  EXPECT_THAT(processor->MessageToTemplateInput(message),
              IsOkAndHolds(nlohmann::ordered_json::parse(R"json({
                "role": "tool",
                "content": [
                  {
                    "type": "text",
                    "text": "{\"key1\": \"value1\", \"key2\": \"value2\"}"
                  }
                ]
              })json")));
}

TEST(Gemma3DataProcessorTest, MessageToTemplateInputWithMultipleToolResponses) {
  ASSERT_OK_AND_ASSIGN(auto processor, Gemma3DataProcessor::Create());
  const nlohmann::ordered_json message = nlohmann::ordered_json::parse(R"json({
    "role": "tool",
    "content": [
      {
        "type": "tool_response",
        "tool_response": {
          "key1": "value1",
          "key2": "value2"
        }
      },
      {
        "type": "tool_response",
        "tool_response": {
          "key3": "value3",
          "key4": "value4"
        }
      }
    ]
  })json");

  // The template input should contain a tool_outputs item with the tool
  // responses formatted as a Python dict.
  EXPECT_THAT(processor->MessageToTemplateInput(message),
              IsOkAndHolds(nlohmann::ordered_json::parse(R"json({
                "role": "tool",
                "content": [
                  {
                    "type": "text",
                    "text": "{\"key1\": \"value1\", \"key2\": \"value2\"}"
                  },
                  {
                    "type": "text",
                    "text": "{\"key3\": \"value3\", \"key4\": \"value4\"}"
                  }
                ]
              })json")));
}

TEST(Gemma3DataProcessorTest, RenderTemplateWithToolCalls) {
  // Load the prompt template.
  const std::string test_file_path =
      GetTestdataPath("google-gemma-3n-e2b-it-tools.jinja");
  ASSERT_OK_AND_ASSIGN(const std::string template_content,
                       GetContents(test_file_path));
  PromptTemplate prompt_template(template_content);

  // Create the message history.
  const nlohmann::ordered_json messages = nlohmann::ordered_json::parse(R"json([
    {
      "role": "user",
      "content":[
        {
          "type": "text",
          "text": "How is the weather in Paris and London?"
        }
      ]
    },
    {
      "role": "assistant",
      "tool_calls": [
        {
          "type": "function",
          "function": {
            "name": "get_weather",
            "arguments": {
              "location": "Paris"
            }
          }
        },
        {
          "type": "function",
          "function": {
            "name": "get_weather",
            "arguments": {
              "location": "London"
            }
          }
        }
      ]
    },
    {
      "role": "tool",
      "content": [
        {
          "type": "tool_response",
          "tool_response": {
            "location": "Paris",
            "temperature": 20,
            "unit": "C",
            "weather": "Sunny"
          }
        },
        {
          "type": "tool_response",
          "tool_response": {
            "location": "London",
            "temperature": 15,
            "unit": "C",
            "weather": "Cloudy"
          }
        }
      ]
    }
  ])json");

  // Create the model data processor.
  ASSERT_OK_AND_ASSIGN(auto processor, Gemma3DataProcessor::Create());

  // Convert the messages to template inputs.
  nlohmann::ordered_json message_template_input =
      nlohmann::ordered_json::array();
  for (const auto& message : messages) {
    ASSERT_OK_AND_ASSIGN(nlohmann::ordered_json input,
                         processor->MessageToTemplateInput(message));
    message_template_input.push_back(input);
  }

  // Render the template.
  PromptTemplateInput template_input = {.messages = message_template_input,
                                        .add_generation_prompt = true};
  ASSERT_OK_AND_ASSIGN(const std::string rendered_prompt,
                       prompt_template.Apply(template_input));

  EXPECT_EQ(rendered_prompt, R"(<start_of_turn>user
How is the weather in Paris and London?<end_of_turn>
<start_of_turn>model
```tool_code
get_weather(location="Paris")
get_weather(location="London")
```<end_of_turn>
<start_of_turn>user
```tool_outputs
{"location": "Paris", "temperature": 20, "unit": "C", "weather": "Sunny"}
{"location": "London", "temperature": 15, "unit": "C", "weather": "Cloudy"}
```<end_of_turn>
<start_of_turn>model
)");
}

// TODO(b/441514829): Enable the tests on Windows once the bug is fixed.
#if !defined(WIN32) && !defined(_WIN32) && !defined(__WIN32__) && \
    !defined(__NT__) && !defined(_WIN64)
TEST(Gemma3DataProcessorTest, ToInputDataVectorTextAndAudio) {
  ASSERT_OK_AND_ASSIGN(auto processor, Gemma3DataProcessor::Create());
  const std::string rendered_template_prompt =
      "<start_of_turn>user\nHere is an audio. Please transcribe it: "
      "<start_of_audio><end_of_turn>";

  std::string audio_path = (std::filesystem::path(::testing::SrcDir()) /
                            kTestdataDir / "audio_sample.wav")
                               .string();
  const nlohmann::ordered_json message = {
      {"role", "user"},
      {"content",
       {{{"type", "text"},
         {"text", "Here is an audio. Please transcribe it: "}},
        {{"type", "audio"}, {"path", audio_path}}}}};
  ASSERT_OK_AND_ASSIGN(
      const std::vector<InputData> input_data,
      processor->ToInputDataVector(rendered_template_prompt,
                                   json::array({message}), {}));

  InputText expected_text1(
      "<start_of_turn>user\nHere is an audio. Please transcribe it: "
      "<start_of_audio>");
  ASSERT_OK_AND_ASSIGN(auto audio_preprocessor,
                       AudioPreprocessorMiniAudio::Create(
                           AudioPreprocessorConfig::CreateDefaultUsmConfig()));
  ASSERT_OK_AND_ASSIGN(
      InputAudio expected_audio,
      audio_preprocessor->Preprocess(InputAudio(ReadFile(audio_path))));
  InputText expected_text2("<end_of_turn>");
  EXPECT_THAT(input_data, ElementsAre(HasInputText(&expected_text1),
                                      HasInputAudio(&expected_audio),
                                      HasInputText(&expected_text2)));
}

TEST(Gemma3DataProcessorTest, PromptTemplateToInputDataVectorTextAndAudio) {
  const std::string test_file_path =
      GetTestdataPath("google-gemma-3n-e2b-it.jinja");
  ASSERT_OK_AND_ASSIGN(const std::string template_content,
                       GetContents(test_file_path));
  PromptTemplate prompt_template(template_content);

  std::string audio_path = (std::filesystem::path(::testing::SrcDir()) /
                            kTestdataDir / "audio_sample.wav")
                               .string();
  const nlohmann::ordered_json messages = {
      {{"role", "system"}, {"content", "Hello world!"}},
      {{"role", "user"},
       {"content",
        {{{"type", "text"}, {"text", "How are you?"}},
         {{"type", "audio"}, {"path", audio_path}}}}},
      {{"role", "assistant"},
       {"content", "I am doing well, thanks for asking."}},
      {{"role", "user"},
       {"content",
        {{{"type", "audio"}, {"path", audio_path}},
         {{"type", "text"}, {"text", "What is the capital of France?"}}}}}};
  PromptTemplateInput template_input = {.messages = messages,
                                        .add_generation_prompt = true};

  ASSERT_OK_AND_ASSIGN(const std::string rendered_prompt,
                       prompt_template.Apply(template_input));

  ASSERT_OK_AND_ASSIGN(auto processor, Gemma3DataProcessor::Create());
  ASSERT_OK_AND_ASSIGN(
      const std::vector<InputData> input_data,
      processor->ToInputDataVector(rendered_prompt, messages, {}));
  InputText expected_text1(R"""(<start_of_turn>user
Hello world!

How are you?<start_of_audio>)""");
  ASSERT_OK_AND_ASSIGN(auto audio_preprocessor,
                       AudioPreprocessorMiniAudio::Create(
                           AudioPreprocessorConfig::CreateDefaultUsmConfig()));
  ASSERT_OK_AND_ASSIGN(
      InputAudio expected_audio,
      audio_preprocessor->Preprocess(InputAudio(ReadFile(audio_path))));
  InputText expected_text2(R"""(<end_of_turn>
<start_of_turn>model
I am doing well, thanks for asking.<end_of_turn>
<start_of_turn>user
<start_of_audio>)""");
  InputText expected_text3(R"""(What is the capital of France?<end_of_turn>
<start_of_turn>model
)""");
  EXPECT_THAT(
      input_data,
      ElementsAre(HasInputText(&expected_text1), HasInputAudio(&expected_audio),
                  HasInputText(&expected_text2), HasInputAudio(&expected_audio),
                  HasInputText(&expected_text3)));
}

#endif  // !defined(WIN32) && !defined(_WIN32) && !defined(__WIN32__) &&
        // !defined(__NT__) && !defined(_WIN64)

}  // namespace
}  // namespace litert::lm
