/*
 * Copyright 2025 Google LLC.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.google.ai.edge.litertlm;

/** A wrapper for the native JNI methods. */
public final class LiteRtLmJni {

  static {
    System.loadLibrary("litertlm_jni");
  }

  private LiteRtLmJni() {}

  /**
   * Creates a new LiteRT-LM engine.
   *
   * @param modelPath The path to the model file.
   * @param backend The backend to use for the engine. It should be the string of the corresponding
   *     value in `litert::lm::Backend`.
   * @param visionBackend The backend to use for the vision executor. If empty, vision executor will
   *     not be initialized. It should be the string of the corresponding value in
   *     `litert::lm::Backend`.
   * @param audioBackend The backend to use for the audio executor. If empty, audio executor will
   *     not be initialized. It should be the string of the corresponding value in
   *     `litert::lm::Backend`.
   * @param maxNumTokens The maximum number of tokens to be processed by the engine. When
   *     non-positive, use the engine's default.
   * @param enableBenchmark Whether to enable benchmark mode or not.
   * @param cacheDir The directory for cache files.
   * @return A pointer to the native engine instance.
   */
  public static native long nativeCreateEngine(
      String modelPath,
      String backend,
      String visionBackend,
      String audioBackend,
      int maxNumTokens,
      boolean enableBenchmark,
      String cacheDir);

  /**
   * Delete the LiteRT-LM engine.
   *
   * @param enginePointer A pointer to the native engine instance.
   */
  public static native void nativeDeleteEngine(long enginePointer);

  /**
   * Creates a new LiteRT-LM session.
   *
   * @param enginePointer A pointer to the native engine instance.
   * @param samplerConfig The sampler configuration.
   * @return A pointer to the native session instance.
   */
  public static native long nativeCreateSession(long enginePointer, SamplerConfig samplerConfig);

  /**
   * Delete the LiteRT-LM session.
   *
   * @param sessionPointer A pointer to the native session instance.
   */
  public static native void nativeDeleteSession(long sessionPointer);

  /**
   * Runs the prefill step for the given input data.
   *
   * @param sessionPointer A pointer to the native session instance.
   * @param inputData An array of {@link InputData} to be processed by the model.
   * @throws LiteRtLmJniException if the underlying native method fails.
   */
  @SuppressWarnings("AvoidObjectArrays") // Array is simpler for JNI
  public static native void nativeRunPrefill(long sessionPointer, InputData[] inputData);

  /**
   * Runs the decode step.
   *
   * @param sessionPointer A pointer to the native session instance.
   * @return The generated content.
   * @throws LiteRtLmJniException if the underlying native method fails.
   */
  public static native String nativeRunDecode(long sessionPointer);

  /**
   * Generates content from the given input data.
   *
   * @param sessionPointer A pointer to the native session instance.
   * @param inputData An array of {@link InputData} to be processed by the model.
   * @return The generated content.
   */
  @SuppressWarnings("AvoidObjectArrays") // Array is simpler for JNI
  public static native String nativeGenerateContent(long sessionPointer, InputData[] inputData);

  /**
   * Generates content from the given input data in a streaming fashion.
   *
   * <p>The [observer] will only receive callbacks if this method returns normally.
   *
   * @param sessionPointer A pointer to the native session instance.
   * @param inputData An array of {@link InputData} to be processed by the model.
   * @param observer The observer to receive the streaming responses.
   */
  @SuppressWarnings("AvoidObjectArrays") // Array is simpler for JNI
  public static native void nativeGenerateContentStream(
      long sessionPointer, InputData[] inputData, JniInferenceCallbacks callback);

  /**
   * Callbacks for the nativeGenerateContentStream.
   *
   * <p>Keep the data type simple (string) to avoid constructing complex JVM object in native layer.
   */
  interface JniInferenceCallbacks {
    /**
     * Called when a new response is generated.
     *
     * @param response The response string.
     */
    void onNext(String response);

    /** Called when the inference is done and finished successfully. */
    void onDone();

    /**
     * Called when an error occurs.
     *
     * @param statusCode The int value of the underlying Status::code returned.
     * @param message The message.
     */
    void onError(int statusCode, String message);
  }

  /**
   * Cancels the ongoing inference process.
   *
   * @param sessionPointer A pointer to the native session instance.
   */
  public static native void nativeCancelProcess(long sessionPointer);

  /**
   * Gets the benchmark info for the session.
   *
   * @param sessionPointer A pointer to the native session instance.
   * @return The benchmark info.
   * @throws LiteRtLmJniException if the underlying native method fails.
   */
  public static native BenchmarkInfo nativeGetBenchmarkInfo(long sessionPointer);

  /**
   * Creates a new LiteRT-LM conversation.
   *
   * @param enginePointer A pointer to the native engine instance.
   * @param samplerConfig The sampler configuration.
   * @param systemMessageJsonString The system instruction to be used in the conversation.
   * @param toolsDescriptionJsonString A json string of a list of tool definitions (Open API json).
   *     could be used.
   * @return A pointer to the native conversation instance.
   */
  public static native long nativeCreateConversation(
      long enginePointer,
      SamplerConfig samplerConfig,
      String systemMessageJsonString,
      String toolsDescriptionJsonString);

  /**
   * Deletes the LiteRT-LM conversation.
   *
   * @param conversationPointer A pointer to the native conversation instance.
   */
  public static native void nativeDeleteConversation(long conversationPointer);

  /**
   * Send message from the given input data asynchronously.
   *
   * <p>The [callbacks] will only receive callbacks if this method returns normally.
   *
   * @param conversationPointer A pointer to the native conversation instance.
   * @param messageJsonString The message to be processed by the native conversation instance.
   * @param callbacks The callbacks to receive the streaming responses.
   */
  public static native void nativeSendMessageAsync(
      long conversationPointer, String messageJsonString, JniMessageCallbacks callbacks);

  /**
   * Cancels the ongoing conversation process.
   *
   * @param conversationPointer A pointer to the native conversation instance.
   */
  public static native void nativeConversationCancelProcess(long conversationPointer);

  /**
   * Gets the benchmark info for the conversation.
   *
   * @param conversationPointer A pointer to the native conversation instance.
   * @return The benchmark info.
   * @throws LiteRtLmJniException if the underlying native method fails.
   */
  public static native BenchmarkInfo nativeConversationGetBenchmarkInfo(long conversationPointer);

  /**
   * Callbacks for the nativeSendMessageAsync.
   *
   * <p>Keep the data type simple (string) to avoid constructing complex JVM object in native layer.
   */
  interface JniMessageCallbacks {
    /**
     * Called when a message is received.
     *
     * @param messageJsonString The message in JSON string format.
     */
    void onMessage(String messageJsonString);

    /** Called when the message stream is done. */
    void onDone();

    /**
     * Called when an error occurs.
     *
     * @param statusCode The int value of the underlying Status::code returned.
     * @param message The message.
     */
    void onError(int statusCode, String message);
  }
}
