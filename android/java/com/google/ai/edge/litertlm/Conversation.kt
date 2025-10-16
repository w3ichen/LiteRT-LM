/*
 * Copyright 2025 Google LLC
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
package com.google.ai.edge.litertlm

import android.util.Log
import java.util.concurrent.CancellationException
import java.util.concurrent.atomic.AtomicBoolean
import org.json.JSONArray
import org.json.JSONObject

/**
 * Represents a conversation with the LiteRT-LM model.
 *
 * Example usage:
 * ```kotlin
 * // Assuming 'engine' is an instance of LiteRtLm
 * val conversation = engine.createConversation()
 *
 * conversation.sendMessageAsync(
 *   Message.of("Hello world"),
 *   object : MessageCallbacks {
 *     override fun onMessage(message: Message) {
 *       // Handle the streaming response
 *       println("Response: ${message.contents[0] as Content.Text).text}")
 *     }
 *
 *     override fun onDone() {
 *       // Handle the end of the response
 *       println("Done")
 *     }
 *
 *     override fun onError(error: Throwable) {
 *       // Handle any errors
 *       println("Error: ${error.message}")
 *     }
 *   },
 * )
 *
 * // Close the conversation at the end to free the underlying native resource.
 * conversation.close()
 * ```
 *
 * This class is AutoCloseable, so you can use `use` block to ensure resources are released.
 *
 * @property handle The native handle to the conversation object.
 * @property toolManager The ToolManager instance to use for this conversation.
 */
// TODO(b/447439217): Support multi-modality.
class Conversation(private val handle: Long, val toolManager: ToolManager) : AutoCloseable {
  private val _isAlive = AtomicBoolean(true)

  /** Whether the conversation is alive and ready to be used, */
  val isAlive: Boolean
    get() = _isAlive.get()

  /**
   * Send message from the given contents in a async fashion.
   *
   * @param contents The list of {@link Content} to be processed by the model.
   * @param callbacks The callbacks to receive the streaming responses.
   * @throws IllegalStateException if the conversation has already been closed or the content is
   *   empty.
   */
  fun sendMessageAsync(message: Message, callbacks: MessageCallbacks) {
    checkIsAlive()

    val jniCallbacks = JniMessageCallbacksImpl(callbacks)
    val messageJSONObject = JSONObject().put("role", "user").put("content", message.toJson())
    LiteRtLmJni.nativeSendMessageAsync(handle, messageJSONObject.toString(), jniCallbacks)
  }

  private inner class JniMessageCallbacksImpl(private val callbacks: MessageCallbacks) :
    LiteRtLmJni.JniMessageCallbacks {

    /** The tool response to be returned back */
    private var pendingToolResponseJSONMessage: JSONObject? = null

    override fun onMessage(messageJsonString: String) {
      val messageJsonObject = JSONObject(messageJsonString)

      if (messageJsonObject.has("tool_calls")) {
        val toolCallsJSONArray = messageJsonObject.getJSONArray("tool_calls")
        val toolResponsesJSONArray = JSONArray()

        for (i in 0..<toolCallsJSONArray.length()) {
          val toolCallJSONObject = toolCallsJSONArray.getJSONObject(i)
          if (!toolCallJSONObject.has("function")) {
            continue
          }
          val functionJSONObject = toolCallJSONObject.getJSONObject("function")
          val functionName = functionJSONObject.getString("name")
          val arguments = functionJSONObject.getJSONObject("arguments")

          Log.i(TAG, "onMessage: Calling tools ${functionName}")
          val result = toolManager.execute(functionName, arguments)
          val toolResponseJSONObject =
            JSONObject()
              .put("type", "tool_response")
              .put("tool_response", JSONObject().put("name", functionName).put("value", result))
          toolResponsesJSONArray.put(toolResponseJSONObject)
        }

        pendingToolResponseJSONMessage =
          JSONObject().put("role", "tool").put("content", toolResponsesJSONArray)
      } else if (messageJsonObject.has("content")) {
        val contentsJsonArray = messageJsonObject.getJSONArray("content")
        val contents = mutableListOf<Content>()

        for (i in 0..<contentsJsonArray.length()) {
          val contentJsonObject = contentsJsonArray.getJSONObject(i)
          val type = contentJsonObject.getString("type")

          if (type == "text") {
            contents.add(Content.Text(contentJsonObject.getString("text")))
          } else {
            Log.w(TAG, "onMessage: Got unsupported content type: $type")
          }
        }
        callbacks.onMessage(Message.of(contents))
      }
    }

    override fun onDone() {
      Log.d(TAG, "onDone")
      val localToolResponse = pendingToolResponseJSONMessage
      if (localToolResponse != null) {
        // If there is pending tool response message, send the message.
        Log.d(TAG, "onDone: Sending tool response.")
        LiteRtLmJni.nativeSendMessageAsync(
          handle,
          localToolResponse.toString(),
          this@JniMessageCallbacksImpl,
        )
        Log.d(TAG, "onDone: Tool response sent.")
        pendingToolResponseJSONMessage = null // Clear after sending
      } else {
        // If no pending action, then call onDone to the original user callbacks.
        callbacks.onDone()
      }
    }

    override fun onError(statusCode: Int, message: String) {
      Log.d(TAG, "onError: $statusCode, $message")

      if (statusCode == 1) { // StatusCode::kCancelled
        callbacks.onError(CancellationException(message))
      } else {
        callbacks.onError(LiteRtLmJniException("Status Code: $statusCode. Message: $message"))
      }
    }
  }

  /**
   * Cancels any ongoing inference process.
   *
   * If there is no ongoing inference process, it is a no-op.
   *
   * @throws IllegalStateException if the session is not alive.
   */
  // b/450903294 is a pending feature request to roll the internal state back.
  fun cancelProcess() {
    checkIsAlive()
    LiteRtLmJni.nativeConversationCancelProcess(handle)
  }

  /**
   * Gets the benchmark info for the conversation.
   *
   * @return The benchmark info.
   * @throws IllegalStateException if the conversation is not alive.
   * @throws LiteRtLmJniException if benchmark is not enabled in the engine config.
   */
  fun getBenchmarkInfo(): BenchmarkInfo {
    checkIsAlive()
    return LiteRtLmJni.nativeConversationGetBenchmarkInfo(handle)
  }

  /**
   * Closes the conversation and releases the native conversation's resources.
   *
   * @throws IllegalStateException if the conversation has already been closed.
   */
  override fun close() {
    if (_isAlive.compareAndSet(true, false)) {
      LiteRtLmJni.nativeDeleteConversation(handle)
    } else {
      throw IllegalStateException("Conversation is closed already.")
    }
  }

  /** Throws [IllegalStateException] if the conversation is not alive. */
  private fun checkIsAlive() {
    check(isAlive) { "Conversation is not alive." }
  }

  companion object {
    private const val TAG = "Conversation"
  }
}

/** An observer for receiving streaming message responses. */
interface MessageCallbacks {
  /**
   * Called when a new message chunk is available from the model.
   *
   * This method may be called multiple times for a single `sendMessageAsync` call as the model
   * streams its response.
   *
   * @param contents The message chunk.
   */
  fun onMessage(message: Message)

  /**
   * Called when all message chunks are sent for a given `sendMessageAsync` call.
   *
   * If the model response includes tool calls, this method is called *after* the tool calls have
   * been executed and their responses have been sent back to the model.
   */
  fun onDone()

  /**
   * Called when an error occurs during the response streaming process.
   *
   * @param throwable The error that occurred.
   */
  fun onError(throwable: Throwable)
}
