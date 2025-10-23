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

/**
 * Experimental flags for the LiteRT-LM.
 *
 * This class is experimental, may change or be removed without notice.
 *
 * To access this APi, the caller need annotation `@OptIn(ExperimentalApi::class)`.
 */
@ExperimentalApi
object ExperimentalFlags {

  /**
   * Whether to force disabling conversation constrained decoding.
   *
   * Note: This flag is read only when a new [Conversation] is created. Changing this value will not
   * affect any existing [Conversation] instances.
   */
  var forceDisableConversationConstrainedDecoding: Boolean = false
}

// Mark this annotation itself as requiring opt-in
@RequiresOptIn(
  message = "This API is experimental and temporary. It may change or be removed without notice.",
  level = RequiresOptIn.Level.WARNING,
)
@Retention(AnnotationRetention.BINARY)
@Target(AnnotationTarget.CLASS, AnnotationTarget.FUNCTION)
annotation class ExperimentalApi
