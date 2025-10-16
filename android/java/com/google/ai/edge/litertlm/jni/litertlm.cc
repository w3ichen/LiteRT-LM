// Copyright 2025 Google LLC.
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

#include <jni.h>
#include <sys/stat.h>

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "nlohmann/json_fwd.hpp"  // from @nlohmann_json
#include "runtime/conversation/conversation.h"
#include "runtime/conversation/io_types.h"
#include "runtime/engine/engine.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/engine/io_types.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/proto/sampler_params.pb.h"

#define JNI_METHOD(METHOD_NAME) \
  Java_com_google_ai_edge_litertlm_LiteRtLmJni_##METHOD_NAME

namespace {
using litert::lm::Backend;
using litert::lm::Conversation;
using litert::lm::ConversationConfig;
using litert::lm::Engine;
using litert::lm::EngineSettings;
using litert::lm::InferenceCallbacks;
using litert::lm::InputAudio;
using litert::lm::InputData;
using litert::lm::InputImage;
using litert::lm::InputText;
using litert::lm::JsonMessage;
using litert::lm::JsonPreface;
using litert::lm::Message;
using litert::lm::ModelAssets;
using litert::lm::Preface;
using litert::lm::Responses;
using litert::lm::SessionConfig;
using litert::lm::proto::SamplerParameters;

void ThrowLiteRtLmJniException(JNIEnv* env, const std::string& message) {
  jclass exClass =
      env->FindClass("com/google/ai/edge/litertlm/LiteRtLmJniException");
  if (exClass != nullptr) {
    env->ThrowNew(exClass, message.c_str());
    // Clean up local reference
    env->DeleteLocalRef(exClass);
  }
}

// Helper function to convert BenchmarkInfo to Java object
jobject CreateBenchmarkInfoJni(
    JNIEnv* env, const litert::lm::BenchmarkInfo& benchmark_info) {
  int last_prefill_token_count = 0;
  if (benchmark_info.GetTotalPrefillTurns() > 0) {
    last_prefill_token_count =
        benchmark_info.GetPrefillTurn(benchmark_info.GetTotalPrefillTurns() - 1)
            .num_tokens;
  }

  int last_decode_token_count = 0;
  if (benchmark_info.GetTotalDecodeTurns() > 0) {
    last_decode_token_count =
        benchmark_info.GetDecodeTurn(benchmark_info.GetTotalDecodeTurns() - 1)
            .num_tokens;
  }

  jclass benchmark_info_cls =
      env->FindClass("com/google/ai/edge/litertlm/BenchmarkInfo");
  jmethodID benchmark_info_ctor =
      env->GetMethodID(benchmark_info_cls, "<init>", "(II)V");

  return env->NewObject(benchmark_info_cls, benchmark_info_ctor,
                        last_prefill_token_count, last_decode_token_count);
}

// Converts a Java InputData array to a C++ vector of InputData.
std::vector<InputData> GetNativeInputData(JNIEnv* env,
                                          jobjectArray input_data) {
  jclass text_class =
      env->FindClass("com/google/ai/edge/litertlm/InputData$Text");
  jclass audio_class =
      env->FindClass("com/google/ai/edge/litertlm/InputData$Audio");
  jclass image_class =
      env->FindClass("com/google/ai/edge/litertlm/InputData$Image");

  jmethodID text_get_text_mid =
      env->GetMethodID(text_class, "getText", "()Ljava/lang/String;");
  jmethodID audio_get_bytes_mid =
      env->GetMethodID(audio_class, "getBytes", "()[B");
  jmethodID image_get_bytes_mid =
      env->GetMethodID(image_class, "getBytes", "()[B");

  jsize num_inputs = env->GetArrayLength(input_data);
  std::vector<InputData> contents;
  contents.reserve(num_inputs);
  for (jsize i = 0; i < num_inputs; ++i) {
    jobject input_obj = env->GetObjectArrayElement(input_data, i);
    if (env->IsInstanceOf(input_obj, text_class)) {
      jstring text_jstr =
          (jstring)env->CallObjectMethod(input_obj, text_get_text_mid);
      const char* text_chars = env->GetStringUTFChars(text_jstr, nullptr);
      contents.emplace_back(InputText(text_chars));
      env->ReleaseStringUTFChars(text_jstr, text_chars);
      env->DeleteLocalRef(text_jstr);
    } else if (env->IsInstanceOf(input_obj, audio_class)) {
      jbyteArray bytes_jarr =
          (jbyteArray)env->CallObjectMethod(input_obj, audio_get_bytes_mid);
      jsize len = env->GetArrayLength(bytes_jarr);
      jbyte* bytes = env->GetByteArrayElements(bytes_jarr, nullptr);
      contents.emplace_back(
          InputAudio(std::string(reinterpret_cast<char*>(bytes), len)));
      env->ReleaseByteArrayElements(bytes_jarr, bytes, JNI_ABORT);
      env->DeleteLocalRef(bytes_jarr);
    } else if (env->IsInstanceOf(input_obj, image_class)) {
      jbyteArray bytes_jarr =
          (jbyteArray)env->CallObjectMethod(input_obj, image_get_bytes_mid);
      jsize len = env->GetArrayLength(bytes_jarr);
      jbyte* bytes = env->GetByteArrayElements(bytes_jarr, nullptr);
      contents.emplace_back(
          InputImage(std::string(reinterpret_cast<char*>(bytes), len)));
      env->ReleaseByteArrayElements(bytes_jarr, bytes, JNI_ABORT);
      env->DeleteLocalRef(bytes_jarr);
    } else {
      ThrowLiteRtLmJniException(env, "Unsupported InputData type");
    }
    env->DeleteLocalRef(input_obj);
  }

  env->DeleteLocalRef(text_class);
  env->DeleteLocalRef(audio_class);
  env->DeleteLocalRef(image_class);

  return contents;
}

class JniInferenceCallbacks : public InferenceCallbacks {
 public:
  JniInferenceCallbacks(JNIEnv* env, JavaVM* jvm, jobject callbacks) {
    callbacks_ = env->NewGlobalRef(callbacks);
    jclass callbacks_class = env->GetObjectClass(callbacks_);
    on_response_mid_ =
        env->GetMethodID(callbacks_class, "onNext", "(Ljava/lang/String;)V");
    on_done_mid_ = env->GetMethodID(callbacks_class, "onDone", "()V");
    on_error_mid_ =
        env->GetMethodID(callbacks_class, "onError", "(ILjava/lang/String;)V");
    env->DeleteLocalRef(callbacks_class);
    jvm_ = jvm;
  }

  ~JniInferenceCallbacks() override {
    bool attached = false;
    JNIEnv* env = GetJniEnvAndAttach(&attached);
    if (env) {
      if (callbacks_) env->DeleteGlobalRef(callbacks_);

      // Detach if attached
      if (attached && jvm_->DetachCurrentThread() != JNI_OK) {
        ABSL_LOG(ERROR)
            << "Failed to detach from JVM in ~JniInferenceCallbacks.";
      }
    } else {
      ABSL_LOG(ERROR)
          << "Failed to get JNIEnv in ~JniInferenceCallbacks, global refs "
             "might be leaked.";
    }
  }

  void OnNext(const Responses& responses) override {
    bool attached = false;
    JNIEnv* env = GetJniEnvAndAttach(&attached);
    if (!env) return;

    auto response_text = responses.GetResponseTextAt(0);
    if (!response_text.ok()) {
      OnError(response_text.status());
    } else {
      jstring response_jstr =
          env->NewStringUTF(std::string(*response_text).c_str());
      env->CallVoidMethod(callbacks_, on_response_mid_, response_jstr);
      env->DeleteLocalRef(response_jstr);
    }

    if (attached && jvm_->DetachCurrentThread() != JNI_OK) {
      ABSL_LOG(ERROR)
          << "Failed to detach from JVM in JniInferenceCallbacks::OnNext.";
    }
  }

  void OnDone() override {
    ABSL_LOG(INFO) << "Receive callback OnDone.";

    bool attached = false;
    JNIEnv* env = GetJniEnvAndAttach(&attached);
    if (env) {
      env->CallVoidMethod(callbacks_, on_done_mid_);
      if (attached && jvm_->DetachCurrentThread() != JNI_OK) {
        ABSL_LOG(ERROR)
            << "Failed to detach from JVM in JniInferenceCallbacks::OnDone.";
      }
    }
  }

  void OnError(const absl::Status& status) override {
    ABSL_LOG(WARNING) << "Receive callback OnError: " << status;

    bool attached = false;
    JNIEnv* env = GetJniEnvAndAttach(&attached);
    if (env) {
      jstring message = env->NewStringUTF(status.message().data());
      env->CallVoidMethod(callbacks_, on_error_mid_, (jint)status.code(),
                          message);
      env->DeleteLocalRef(message);

      if (attached && jvm_->DetachCurrentThread() != JNI_OK) {
        ABSL_LOG(ERROR)
            << "Failed to detach from JVM in JniInferenceCallbacks::OnError.";
      }
    }
  }

 private:
  JNIEnv* GetJniEnvAndAttach(bool* attached) {
    JNIEnv* env = nullptr;
    *attached = false;
    int get_env_stat = jvm_->GetEnv((void**)&env, JNI_VERSION_1_6);
    if (get_env_stat == JNI_EDETACHED) {
#if defined(__ANDROID__)
      if (jvm_->AttachCurrentThread(&env, nullptr) == 0) {
#else
      if (jvm_->AttachCurrentThread((void**)&env, nullptr) == 0) {
#endif
        *attached = true;
        return env;
      } else {
        ABSL_LOG(ERROR) << "Failed to attach to JVM.";
        return nullptr;
      }
    } else if (get_env_stat == JNI_OK) {
      return env;
    } else {
      ABSL_LOG(ERROR) << "Failed to get JNIEnv: GetEnv returned "
                      << get_env_stat;
      return nullptr;
    }
  }

  JavaVM* jvm_;
  jobject callbacks_;
  jmethodID on_response_mid_;
  jmethodID on_done_mid_;
  jmethodID on_error_mid_;
};

class JniMessageCallbacks : public litert::lm::MessageCallbacks {
 public:
  JniMessageCallbacks(JNIEnv* env, JavaVM* jvm, jobject callbacks) {
    callbacks_ = env->NewGlobalRef(callbacks);
    jclass callbacks_class = env->GetObjectClass(callbacks_);
    on_message_mid_ =
        env->GetMethodID(callbacks_class, "onMessage", "(Ljava/lang/String;)V");
    on_complete_mid_ = env->GetMethodID(callbacks_class, "onDone", "()V");
    on_error_mid_ =
        env->GetMethodID(callbacks_class, "onError", "(ILjava/lang/String;)V");
    env->DeleteLocalRef(callbacks_class);
    jvm_ = jvm;
  }

  ~JniMessageCallbacks() override {
    bool attached = false;
    JNIEnv* env = GetJniEnvAndAttach(&attached);
    if (env) {
      if (callbacks_) env->DeleteGlobalRef(callbacks_);

      // Detach if attached
      if (attached && jvm_->DetachCurrentThread() != JNI_OK) {
        ABSL_LOG(ERROR) << "Failed to detach from JVM in ~JniMessageCallbacks.";
      }
    } else {
      ABSL_LOG(ERROR)
          << "Failed to get JNIEnv in ~JniMessageCallbacks, global refs "
             "might be leaked.";
    }
  }

  void OnMessage(const Message& message) override {
    bool attached = false;
    JNIEnv* env = GetJniEnvAndAttach(&attached);
    if (!env) return;

    if (!std::holds_alternative<litert::lm::JsonMessage>(message)) {
      OnError(absl::InvalidArgumentError("Json message is required for now."));
    } else {
      auto json_message = std::get<litert::lm::JsonMessage>(message);
      std::string message_str = json_message.dump();
      jstring message_jstr = env->NewStringUTF(message_str.c_str());
      env->CallVoidMethod(callbacks_, on_message_mid_, message_jstr);
      env->DeleteLocalRef(message_jstr);
    }

    if (attached && jvm_->DetachCurrentThread() != JNI_OK) {
      ABSL_LOG(ERROR)
          << "Failed to detach from JVM in JniMessageCallbacks::OnMessage.";
    }
  }

  void OnComplete() override {
    bool attached = false;
    JNIEnv* env = GetJniEnvAndAttach(&attached);
    if (env) {
      env->CallVoidMethod(callbacks_, on_complete_mid_);
      if (attached && jvm_->DetachCurrentThread() != JNI_OK) {
        ABSL_LOG(ERROR)
            << "Failed to detach from JVM in JniMessageCallbacks::OnComplete.";
      }
    }
  }

  void OnError(const absl::Status& status) override {
    ABSL_LOG(WARNING) << "Receive callback OnError: " << status;

    bool attached = false;
    JNIEnv* env = GetJniEnvAndAttach(&attached);
    if (env) {
      jstring message = env->NewStringUTF(status.message().data());
      env->CallVoidMethod(callbacks_, on_error_mid_, (jint)status.code(),
                          message);
      env->DeleteLocalRef(message);
      if (attached && jvm_->DetachCurrentThread() != JNI_OK) {
        ABSL_LOG(ERROR)
            << "Failed to detach from JVM in JniMessageCallbacks::OnError.";
      }
    }
  }

 private:
  JNIEnv* GetJniEnvAndAttach(bool* attached) {
    JNIEnv* env = nullptr;
    *attached = false;
    int get_env_stat = jvm_->GetEnv((void**)&env, JNI_VERSION_1_6);
    if (get_env_stat == JNI_EDETACHED) {
#if defined(__ANDROID__)
      if (jvm_->AttachCurrentThread(&env, nullptr) == 0) {
#else
      if (jvm_->AttachCurrentThread((void**)&env, nullptr) == 0) {
#endif
        *attached = true;
        return env;
      } else {
        ABSL_LOG(ERROR) << "Failed to attach to JVM.";
        return nullptr;
      }
    } else if (get_env_stat == JNI_OK) {
      return env;
    } else {
      ABSL_LOG(ERROR) << "Failed to get JNIEnv: GetEnv returned "
                      << get_env_stat;
      return nullptr;
    }
  }

  JavaVM* jvm_;
  jobject callbacks_;
  jmethodID on_message_mid_;
  jmethodID on_complete_mid_;
  jmethodID on_error_mid_;
};

// Helper function to create SamplerParameters from Java SamplerConfig object.
SamplerParameters CreateSamplerParamsFromJni(JNIEnv* env,
                                             jobject sampler_config_obj) {
  SamplerParameters sampler_params;

  // Based on the current engine implementation, when the SamplerConfig is
  // set, we must switch to the TOP_P sampling type.
  sampler_params.set_type(SamplerParameters::TOP_P);

  // Get the fields from SamplerConfig
  jclass sampler_config_cls = env->GetObjectClass(sampler_config_obj);

  jmethodID get_top_k_mid =
      env->GetMethodID(sampler_config_cls, "getTopK", "()I");
  sampler_params.set_k(env->CallIntMethod(sampler_config_obj, get_top_k_mid));

  jmethodID get_top_p_mid =
      env->GetMethodID(sampler_config_cls, "getTopP", "()D");
  sampler_params.set_p(
      env->CallDoubleMethod(sampler_config_obj, get_top_p_mid));

  jmethodID get_temperature_mid =
      env->GetMethodID(sampler_config_cls, "getTemperature", "()D");
  sampler_params.set_temperature(
      env->CallDoubleMethod(sampler_config_obj, get_temperature_mid));

  jmethodID get_seed_mid =
      env->GetMethodID(sampler_config_cls, "getSeed", "()I");
  sampler_params.set_seed(env->CallIntMethod(sampler_config_obj, get_seed_mid));

  env->DeleteLocalRef(sampler_config_cls);

  return sampler_params;
}

}  // namespace

extern "C" {

JNIEXPORT jlong JNICALL JNI_METHOD(nativeCreateEngine)(
    JNIEnv* env, jclass thiz, jstring model_path, jstring backend,
    jstring vision_backend, jstring audio_backend, jint max_num_tokens,
    jboolean enable_benchmark, jstring cache_dir) {
  const char* model_path_chars = env->GetStringUTFChars(model_path, nullptr);
  std::string model_path_str(model_path_chars);
  env->ReleaseStringUTFChars(model_path, model_path_chars);

  // Check if the file exists.
  struct stat buffer;
  if (stat(model_path_str.c_str(), &buffer) != 0) {
    ThrowLiteRtLmJniException(env, "Model file not found: " + model_path_str);
    return 0;
  }

  auto model_assets = ModelAssets::Create(model_path_str);
  if (!model_assets.ok()) {
    ThrowLiteRtLmJniException(env, "Failed to create model assets: " +
                                       model_assets.status().ToString());
    return 0;
  }

  const char* backend_chars = env->GetStringUTFChars(backend, nullptr);
  std::string backend_str(backend_chars);
  env->ReleaseStringUTFChars(backend, backend_chars);

  auto backend_enum = litert::lm::GetBackendFromString(backend_str);
  if (!backend_enum.ok()) {
    ThrowLiteRtLmJniException(env, backend_enum.status().ToString());
    return 0;
  }

  const char* vision_backend_chars =
      env->GetStringUTFChars(vision_backend, nullptr);
  std::string vision_backend_str(vision_backend_chars);
  env->ReleaseStringUTFChars(vision_backend, vision_backend_chars);

  std::optional<Backend> vision_backend_optional = std::nullopt;
  if (!vision_backend_str.empty()) {
    auto vision_backend_enum =
        litert::lm::GetBackendFromString(vision_backend_str);
    if (!vision_backend_enum.ok()) {
      ThrowLiteRtLmJniException(env, vision_backend_enum.status().ToString());
      return 0;
    }

    vision_backend_optional = vision_backend_enum.value();
  }

  const char* audio_backend_chars =
      env->GetStringUTFChars(audio_backend, nullptr);
  std::string audio_backend_str(audio_backend_chars);
  env->ReleaseStringUTFChars(audio_backend, audio_backend_chars);

  std::optional<Backend> audio_backend_optional = std::nullopt;
  if (!audio_backend_str.empty()) {
    auto audio_backend_enum =
        litert::lm::GetBackendFromString(audio_backend_str);
    if (!audio_backend_enum.ok()) {
      ThrowLiteRtLmJniException(env, audio_backend_enum.status().ToString());
      return 0;
    }

    audio_backend_optional = audio_backend_enum.value();
  }

  auto settings = EngineSettings::CreateDefault(*model_assets, *backend_enum,
                                                vision_backend_optional,
                                                audio_backend_optional);
  if (!settings.ok()) {
    ThrowLiteRtLmJniException(env, "Failed to create engine settings: " +
                                       settings.status().ToString());
    return 0;
  }

  const char* cache_dir_chars = env->GetStringUTFChars(cache_dir, nullptr);
  std::string cache_dir_str(cache_dir_chars);
  env->ReleaseStringUTFChars(cache_dir, cache_dir_chars);
  if (!cache_dir_str.empty()) {
    settings->GetMutableMainExecutorSettings().SetCacheDir(cache_dir_str);
  }

  if (max_num_tokens > 0) {
    settings->GetMutableMainExecutorSettings().SetMaxNumTokens(max_num_tokens);
  }

  if (enable_benchmark) {
    settings->GetMutableBenchmarkParams();
  }

  auto engine = Engine::CreateEngine(*settings);
  if (!engine.ok()) {
    ThrowLiteRtLmJniException(
        env, "Failed to create engine: " + engine.status().ToString());
    return 0;
  }

  return reinterpret_cast<jlong>(engine->release());
}

JNIEXPORT void JNICALL JNI_METHOD(nativeDeleteEngine)(JNIEnv* env, jclass thiz,
                                                      jlong engine_pointer) {
  delete reinterpret_cast<Engine*>(engine_pointer);
}

JNIEXPORT jlong JNICALL
JNI_METHOD(nativeCreateSession)(JNIEnv* env, jclass thiz, jlong engine_pointer,
                                jobject sampler_config_obj) {
  auto session_config = SessionConfig::CreateDefault();

  if (sampler_config_obj != nullptr) {
    session_config.GetMutableSamplerParams() =
        CreateSamplerParamsFromJni(env, sampler_config_obj);
  }

  Engine* engine = reinterpret_cast<Engine*>(engine_pointer);
  auto session = engine->CreateSession(session_config);
  if (!session.ok()) {
    ThrowLiteRtLmJniException(
        env, "Failed to create session: " + session.status().ToString());
    return 0;
  }
  return reinterpret_cast<jlong>(session->release());
}

JNIEXPORT void JNICALL JNI_METHOD(nativeDeleteSession)(JNIEnv* env, jclass thiz,
                                                       jlong session_pointer) {
  delete reinterpret_cast<Engine::Session*>(session_pointer);
}

JNIEXPORT void JNICALL JNI_METHOD(nativeRunPrefill)(JNIEnv* env, jclass thiz,
                                                    jlong session_pointer,
                                                    jobjectArray input_data) {
  Engine::Session* session =
      reinterpret_cast<Engine::Session*>(session_pointer);

  std::vector<InputData> contents = GetNativeInputData(env, input_data);
  // return if there is pending exceptions (e.g., if ThrowLiteRtLmJniException
  // called.)
  if (env->ExceptionCheck()) {
    return;
  }

  auto status = session->RunPrefill(contents);

  if (!status.ok()) {
    ThrowLiteRtLmJniException(env,
                              "Failed to run prefill: " + status.ToString());
  }
}

JNIEXPORT jstring JNICALL JNI_METHOD(nativeRunDecode)(JNIEnv* env, jclass thiz,
                                                      jlong session_pointer) {
  Engine::Session* session =
      reinterpret_cast<Engine::Session*>(session_pointer);

  auto responses = session->RunDecode();

  if (!responses.ok()) {
    ThrowLiteRtLmJniException(
        env, "Failed to run decode: " + responses.status().ToString());
    return nullptr;
  }

  if (responses->GetNumOutputCandidates() != 1) {
    ThrowLiteRtLmJniException(
        env, "Number of output candidates should be 1, but got " +
                 std::to_string(responses->GetNumOutputCandidates()));
  }

  auto response_text = responses->GetResponseTextAt(0);
  if (!response_text.ok()) {
    ThrowLiteRtLmJniException(env, "Failed to get response text: " +
                                       response_text.status().ToString());
    return nullptr;
  }

  return env->NewStringUTF(std::string(*response_text).c_str());
}

JNIEXPORT jstring JNICALL JNI_METHOD(nativeGenerateContent)(
    JNIEnv* env, jclass thiz, jlong session_pointer, jobjectArray input_data) {
  Engine::Session* session =
      reinterpret_cast<Engine::Session*>(session_pointer);

  std::vector<InputData> contents = GetNativeInputData(env, input_data);
  // return if there is pending exceptions (e.g., if ThrowLiteRtLmJniException
  // called.)
  if (env->ExceptionCheck()) {
    return nullptr;
  }

  auto responses = session->GenerateContent(contents);

  if (!responses.ok()) {
    ThrowLiteRtLmJniException(
        env, "Failed to generate content: " + responses.status().ToString());
    return nullptr;
  }

  if (responses->GetNumOutputCandidates() == 0) {
    return env->NewStringUTF("");
  }

  auto response_text = responses->GetResponseTextAt(0);
  if (!response_text.ok()) {
    ThrowLiteRtLmJniException(env, "Failed to get response text: " +
                                       response_text.status().ToString());
    return nullptr;
  }

  return env->NewStringUTF(std::string(*response_text).c_str());
}

JNIEXPORT void JNICALL JNI_METHOD(nativeGenerateContentStream)(
    JNIEnv* env, jclass thiz, jlong session_pointer, jobjectArray input_data,
    jobject callbacks) {
  JavaVM* jvm = nullptr;
  if (env->GetJavaVM(&jvm) != JNI_OK) {
    ThrowLiteRtLmJniException(env, "Failed to get JavaVM");
    return;
  }

  Engine::Session* session =
      reinterpret_cast<Engine::Session*>(session_pointer);

  std::vector<InputData> contents = GetNativeInputData(env, input_data);
  if (env->ExceptionCheck()) {
    return;
  }

  auto jni_callbacks =
      std::make_unique<JniInferenceCallbacks>(env, jvm, callbacks);
  auto status =
      session->GenerateContentStream(contents, std::move(jni_callbacks));

  if (!status.ok()) {
    ThrowLiteRtLmJniException(
        env, "Failed to start GenerateContentStream: " + status.ToString());
  }
}

JNIEXPORT void JNICALL JNI_METHOD(nativeCancelProcess)(JNIEnv* env, jclass thiz,
                                                       jlong session_pointer) {
  Engine::Session* session =
      reinterpret_cast<Engine::Session*>(session_pointer);
  session->CancelProcess();
}

JNIEXPORT jobject JNICALL JNI_METHOD(nativeGetBenchmarkInfo)(
    JNIEnv* env, jclass thiz, jlong session_pointer) {
  Engine::Session* session =
      reinterpret_cast<Engine::Session*>(session_pointer);

  auto benchmark_info = session->GetBenchmarkInfo();
  if (!benchmark_info.ok()) {
    ThrowLiteRtLmJniException(env, "Failed to get benchmark info: " +
                                       benchmark_info.status().ToString());
    return nullptr;
  }

  return CreateBenchmarkInfoJni(env, *benchmark_info);
}

JNIEXPORT jobject JNICALL JNI_METHOD(nativeConversationGetBenchmarkInfo)(
    JNIEnv* env, jclass thiz, jlong conversation_pointer) {
  Conversation* conversation =
      reinterpret_cast<Conversation*>(conversation_pointer);

  auto benchmark_info = conversation->GetBenchmarkInfo();
  if (!benchmark_info.ok()) {
    ThrowLiteRtLmJniException(env, "Failed to get benchmark info: " +
                                       benchmark_info.status().ToString());
    return nullptr;
  }

  return CreateBenchmarkInfoJni(env, *benchmark_info);
}

JNIEXPORT jlong JNICALL JNI_METHOD(nativeCreateConversation)(
    JNIEnv* env, jclass thiz, jlong engine_pointer, jobject sampler_config_obj,
    jstring system_message_json_string, jstring tools_description_json_string) {
  Engine* engine = reinterpret_cast<Engine*>(engine_pointer);

  // Create a native SessionConfig
  auto session_config = SessionConfig::CreateDefault();
  if (sampler_config_obj != nullptr) {
    session_config.GetMutableSamplerParams() =
        CreateSamplerParamsFromJni(env, sampler_config_obj);
  }

  // Set an empty user's prefix field to avoid prompt template being
  // overridden by the llm metadata.
  auto emptyTemplate = litert::lm::proto::PromptTemplates();
  auto affixes = emptyTemplate.mutable_user();
  affixes->set_prefix("");
  session_config.GetMutablePromptTemplates() = emptyTemplate;

  // Create the Preface from the system instruction and tools.
  JsonPreface json_preface;

  const char* system_message_chars =
      env->GetStringUTFChars(system_message_json_string, nullptr);
  std::string system_message_json_str(system_message_chars);
  env->ReleaseStringUTFChars(system_message_json_string, system_message_chars);
  if (!system_message_json_str.empty()) {
    nlohmann::ordered_json system_message;
    system_message["role"] = "system";
    system_message["content"] =
        nlohmann::ordered_json::parse(system_message_json_str);

    nlohmann::ordered_json::array_t messages;
    messages.push_back(system_message);
    json_preface.messages = messages;
  }

  const char* tools_description_chars =
      env->GetStringUTFChars(tools_description_json_string, nullptr);
  auto tool_json = nlohmann::ordered_json::parse(tools_description_chars);
  env->ReleaseStringUTFChars(tools_description_json_string,
                             tools_description_chars);

  if (tool_json.is_array()) {
    nlohmann::ordered_json::array_t tool_json_array =
        tool_json.get<nlohmann::ordered_json::array_t>();
    json_preface.tools = tool_json_array;
  } else {
    ThrowLiteRtLmJniException(
        env, "tools_json should be a json array. Got: " + tool_json.dump());
    return 0;
  }

  std::optional<Preface> preface = json_preface;

  // Create the conversation
  auto conversation_config = ConversationConfig::CreateFromSessionConfig(
      *engine, session_config, preface);
  if (!conversation_config.ok()) {
    ThrowLiteRtLmJniException(env, "Failed to create conversation config: " +
                                       conversation_config.status().ToString());
    return 0;
  }
  auto conversation = Conversation::Create(*engine, *conversation_config);
  if (!conversation.ok()) {
    ThrowLiteRtLmJniException(env, "Failed to create conversation: " +
                                       conversation.status().ToString());
    return 0;
  }

  return reinterpret_cast<jlong>(conversation->release());
}

JNIEXPORT void JNICALL JNI_METHOD(nativeDeleteConversation)(
    JNIEnv* env, jclass thiz, jlong conversation_pointer) {
  delete reinterpret_cast<Conversation*>(conversation_pointer);
}

JNIEXPORT void JNICALL JNI_METHOD(nativeSendMessageAsync)(
    JNIEnv* env, jclass thiz, jlong conversation_pointer,
    jstring messageJSONString, jobject callbacks) {
  JavaVM* jvm = nullptr;
  if (env->GetJavaVM(&jvm) != JNI_OK) {
    ThrowLiteRtLmJniException(env, "Failed to get JavaVM");
    return;
  }

  Conversation* conversation =
      reinterpret_cast<Conversation*>(conversation_pointer);

  const char* json_chars = env->GetStringUTFChars(messageJSONString, nullptr);
  litert::lm::JsonMessage json_message =
      nlohmann::ordered_json::parse(json_chars);
  env->ReleaseStringUTFChars(messageJSONString, json_chars);

  auto jni_callbacks =
      std::make_unique<JniMessageCallbacks>(env, jvm, callbacks);
  auto status =
      conversation->SendMessageStream(json_message, std::move(jni_callbacks));

  if (!status.ok()) {
    ThrowLiteRtLmJniException(
        env, "Failed to start nativeSendMessageAsync: " + status.ToString());
  }
}

JNIEXPORT void JNICALL JNI_METHOD(nativeConversationCancelProcess)(
    JNIEnv* env, jclass thiz, jlong conversation_pointer) {
  Conversation* conversation =
      reinterpret_cast<Conversation*>(conversation_pointer);
  conversation->CancelProcess();
}

}  // extern "C"
