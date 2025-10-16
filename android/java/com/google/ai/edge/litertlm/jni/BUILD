# Copyright 2025 Google LLC.
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

DEPS = [
    "@com_google_absl//absl/log:absl_log",
    "@com_google_absl//absl/status",
    "@com_google_absl//absl/status:statusor",
    "@nlohmann_json//:json",
    "//runtime/conversation",
    "//runtime/conversation:io_types",
    "//runtime/core:engine_impl",  # buildcleaner: keep
    "//runtime/engine:engine_interface",
    "//runtime/engine:engine_settings",
    "//runtime/engine:io_types",
    "//runtime/executor:executor_settings_base",
    "//runtime/proto:sampler_params_cc_proto",
]

cc_binary(
    name = "jni",
    srcs = ["litertlm.cc"],
    deps = DEPS,
    linkshared = 1,
    tags = ["manual"], # avoid building without android flags in CI
)
