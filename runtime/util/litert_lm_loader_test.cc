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

#include "runtime/util/litert_lm_loader.h"

#include <filesystem>  // NOLINT: Required for path manipulation.
#include <utility>

#include <gtest/gtest.h>
#include "runtime/components/model_resources.h"
#include "runtime/util/scoped_file.h"

namespace litert::lm {

namespace {

TEST(LitertLmLoaderTest, InitializeWithSentencePieceFile) {
  const auto model_path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/test_lm.litertlm";
  auto model_file = ScopedFile::Open(model_path.string());
  EXPECT_TRUE(model_file.ok());
  LitertLmLoader loader(std::move(model_file.value()));
  EXPECT_FALSE(loader.GetHuggingFaceTokenizer());
  EXPECT_GT(loader.GetSentencePieceTokenizer()->Size(), 0);
  EXPECT_GT(loader.GetTFLiteModel(ModelType::kTfLitePrefillDecode).Size(), 0);
  EXPECT_GT(loader.GetLlmMetadata().Size(), 0);
  // Try to get non-existent TFLite model.
  EXPECT_EQ(loader.GetTFLiteModel(ModelType::kTfLiteEmbedder).Size(), 0);
}

TEST(LitertLmLoaderTest, InitializeWithHuggingFaceFile) {
  const auto model_path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/test_hf_tokenizer.litertlm";
  auto model_file = ScopedFile::Open(model_path.string());
  ASSERT_TRUE(model_file.ok());
  LitertLmLoader loader(std::move(model_file.value()));
  ASSERT_GT(loader.GetHuggingFaceTokenizer()->Size(), 0);
  ASSERT_FALSE(loader.GetSentencePieceTokenizer());
}

}  // namespace
}  // namespace litert::lm
