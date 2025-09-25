# Copyright 2025 The ODML Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# [Google-internal load of `cc_library`]
# [Google-internal load of `cc_test`]

package(
    default_hdrs_check = "strict",
    default_visibility = [
        "//:__subpackages__",
    ],
)

cc_library(
    name = "config_registry",
    hdrs = [
        "config_registry.h",
    ],
    deps = [
        ":gemma3_data_processor_config",
    ],
)

cc_library(
    name = "gemma3_data_processor_config",
    hdrs = [
        "gemma3_data_processor_config.h",
    ],
)

cc_library(
    name = "model_data_processor",
    hdrs = ["model_data_processor.h"],
    deps = [
        ":config_registry",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@nlohmann_json//:json",
        "//runtime/conversation:io_types",
        "//runtime/engine:io_types",
    ],
)

cc_library(
    name = "gemma3_data_processor",
    srcs = ["gemma3_data_processor.cc"],
    hdrs = ["gemma3_data_processor.h"],
    deps = [
        ":gemma3_data_processor_config",
        ":model_data_processor",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@nlohmann_json//:json",
        "//runtime/components/tool_use:parser_utils",
        "//runtime/components/tool_use:python_tool_format_utils",
        "//runtime/conversation:io_types",
        "//runtime/engine:io_types",
        "//runtime/util:litert_status_util",
    ],
)

cc_test(
    name = "gemma3_data_processor_test",
    srcs = ["gemma3_data_processor_test.cc"],
    data = [
        "//runtime/components/testdata",
    ],
    deps = [
        ":gemma3_data_processor",
        ":gemma3_data_processor_config",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@nlohmann_json//:json",
        "//runtime/components:prompt_template",
        "//runtime/conversation:io_types",
        "//runtime/engine:io_types",
        "//runtime/util:test_utils",
    ],
)
