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

import android.util.Base64
import org.json.JSONArray
import org.json.JSONObject

/** Represents a message in the conversation. A message can contain multiple [Content]. */
class Message private constructor(val contents: List<Content>) {

  fun init() {
    check(contents.isNotEmpty()) { "Contents should not be empty." }
  }

  /** Convert to [JSONArray]. Used internally. */
  internal fun toJson(): JSONArray {
    return JSONArray().apply {
      for (content in contents) {
        this.put(content.toJson())
      }
    }
  }

  companion object {
    /** Creates a [Message] from a text string. */
    fun of(text: String): Message {
      return Message(listOf(Content.Text(text)))
    }

    /** Creates a [Message] from a single [Content]. */
    fun of(content: Content): Message {
      return Message(listOf(content))
    }

    /** Creates a [Message] from a list of [Content]. */
    fun of(contents: List<Content>): Message {
      return Message(contents)
    }
  }
}

/** Represents a content in the [Message] of the conversation. */
sealed class Content {
  /** Convert to [JSONObject]. Used internally. */
  internal abstract fun toJson(): JSONObject

  /** Text. */
  data class Text(val text: String) : Content() {
    override fun toJson(): JSONObject {
      return JSONObject().put("type", "text").put("text", text)
    }
  }

  /** Image provided as raw bytes. */
  data class ImageBytes(val bytes: ByteArray) : Content() {
    override fun toJson(): JSONObject {
      return JSONObject()
        .put("type", "image")
        .put("blob", Base64.encodeToString(bytes, Base64.DEFAULT))
    }
  }

  /** Audio provided as raw bytes. */
  data class AudioBytes(val bytes: ByteArray) : Content() {
    override fun toJson(): JSONObject {
      return JSONObject()
        .put("type", "audio")
        .put("blob", Base64.encodeToString(bytes, Base64.DEFAULT))
    }
  }
  // TODO(b/447439217): Add more ways to construct contents, e.g., by file, url.
}
