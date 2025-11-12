# LiteRT-LM Android API

This document provides an overview and usage examples for the LiteRT-LM Android
API, focusing on the Conversation API.

## Overview

The LiteRT-LM Android API allows developers to integrate large language model
capabilities into their Android applications. The API is designed to be flexible
and efficient, supporting various use cases from simple text generation,
multi-modality (audio and vision), and complex conversational interactions with
tool use.

The core components of the Conversation API include:

-   **`Engine`**: Manages the lifecycle of a LiteRT-LM model. It handles loading
    the model and creating conversations.
-   **`EngineConfig`**: Configuration for the `Engine`, including the model
    path, backend (CPU/GPU), and other settings.
-   **`Conversation`**: Represents a conversational session with the model. It
    maintains the state of the conversation and handles message exchanges.
-   **`ConversationConfig`**: Configuration for a `Conversation`, including
    system messages, tools, and sampling parameters.
-   **`Message`**: Represents a single message within a conversation, which can
    contain various types of content.
-   **`Content`**: Sealed class representing different types of content within a
    `Message` (e.g., Text, Image, Audio).
-   **`MessageCallback`**: An interface for handling streaming responses from
    the model asynchronously.
-   **`@Tool` / `@ToolParam`**: Annotations used to define custom functions as
    tools that the model can invoke.
-   **`ToolManager`**: Manages the registration and execution of tools.

## Getting Started

### 1. Add the Gradle dependency

```
dependencies {
    implementation("com.google.ai.edge.litertlm:litertlm:LATEST_VERSION")
}
```

Replace `LATEST_VERSION` with the actual version you intend to use. The
LiteRT-LM Android API package can be found on [Google
Maven](https://maven.google.com/web/index.html#com.google.ai.edge.litertlm:litertlm).

The GPU backend uses some native system libraries. The app need to request
explicitly by adding the following to your `AndroidManifest.xml` inside the
`<application>` tag:

```xml
  <application>
    <uses-native-library android:name="libvndksupport.so" android:required="false"/>
    <uses-native-library android:name="libOpenCL.so" android:required="false"/>
  </application>
```

### 2. Initialize the Engine

The `Engine` is the entry point to the API. Initialize it with the model path
and configuration. Remember to close the engine to release resources.

**Note:** The `engine.initialize()` method can take a significant amount of time
(e.g., up to 10 seconds) to load the model. It is strongly recommended to call
this on a background thread or coroutine to avoid blocking the UI thread.

```kotlin
import com.google.ai.edge.litertlm.Backend
import com.google.ai.edge.litertlm.Engine
import com.google.ai.edge.litertlm.EngineConfig

// Assuming 'context' is your Application Context
val engineConfig = EngineConfig(
    modelPath = "/path/to/your/model.litertlm", // Replace with your model path
    backend = Backend.CPU, // Or Backend.GPU
    // optional: Pick a writable dir. This can improve 2nd load time.
    // cacheDir = "/tmp/" or context.cacheDir.path (for Android)
)

val engine = Engine(engineConfig)
engine.initialize()
// ... Use the engine to create conversation ...

// Close the engine when done
engine.close()
```

### 3. Create a Conversation

Once the engine is initialized, create a `Conversation` instance. You can
provide a `ConversationConfig` to customize its behavior.

```kotlin
import com.google.ai.edge.litertlm.ConversationConfig
import com.google.ai.edge.litertlm.Message
import com.google.ai.edge.litertlm.SamplerConfig

// Optional: Configure system message and sampling parameters
val conversationConfig = ConversationConfig(
    systemMessage = Message.of("You are a helpful assistant."),
    samplerConfig = SamplerConfig(topK = 10, topP = 0.95, temperature = 0.8),
)

val conversation = engine.createConversation(conversationConfig)
// Or with default config:
// val conversation = engine.createConversation()

// ... Use the conversation ...

// Close the conversation when done
conversation.close()
```

`Conversation` implements `AutoCloseable`, so you can use the `use` block for
automatic resource management for one-shot or short-lived conversation:

```kotlin
engine.createConversation(conversationConfig).use { conversation ->
    // Interact with the conversation
}
```

### 4. Sending Messages

There are three ways to send messages:

-   **`sendMessage(message: Message): Message`**: Synchronous call that blocks
    until the model returns a complete response. This is simpler for basic
    request/response interactions.
-   **`sendMessageAsync(message: Message, callback: MessageCallback)`**:
    Asynchronous call for streaming responses. This is better for long-running
    requests or when you want to display the response as it's being
    generated.
-   **`sendMessageAsync(message: Message): Flow<Message>`**: Asynchronous call
    that returns a Kotlin Flow for streaming responses. This is the recommended
    approach for Coroutine users.

**Synchronous Example:**

```kotlin
import com.google.ai.edge.litertlm.Content
import com.google.ai.edge.litertlm.Message

val userMessage = Message.of("What is the capital of France?")
print(conversation.sendMessage(userMessage))
```

**Asynchronous Example with callback:**

Use `sendMessageAsync` to send a message to the model and receive responses
through callback.

```kotlin
import com.google.ai.edge.litertlm.Content
import com.google.ai.edge.litertlm.Message
import com.google.ai.edge.litertlm.MessageCallback
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit

val callback = object : MessageCallback {
    override fun onMessage(message: Message) {
        print(message)
    }

    override fun onDone() {
        // Streaming completed
    }

    override fun onError(throwable: Throwable) {
        // Error during streaming
    }
}

val userMessage = Message.of("What is the capital of France?")
conversation.sendMessageAsync(userMessage, callback)
```

**Asynchronous Example with Flow:**

Use `sendMessageAsync` (without the callback arg) to send a message to the model
and receive responses through a Kotlin Flow.

```kotlin
import com.google.ai.edge.litertlm.Content
import com.google.ai.edge.litertlm.Message
import kotlinx.coroutines.launch

// Within a coroutine scope
val userMessage = Message.of("What is the capital of France?")
conversation.sendMessageAsync(userMessage)
    .catch { ... } // error during streaming
    .collect{ print(it.toString()) }
```

### 5. Multi-Modality

`Message` objects can contain different types of `Content`, including `Text`,
`ImageBytes`, `ImageFile`, and `AudioBytes`, `AudioFile`.

```kotlin
val audioBytes: ByteArray = // Load audio bytes

// See the Content class for other variants.
val multiModalMessage = Message.of(
    Content.ImageFile("/path/to/image"),
    Content.AudioBytes(audioBytes),
    Content.Text("Describe this image and audio."),
)
```

### 6. Defining and Using Tools

You can define custom Kotlin functions as tools that the model can call to
perform actions or fetch information.

#### Defining a ToolSet

Create a class and annotate methods with `@Tool` and parameters with
`@ToolParam`.

```kotlin
import com.google.ai.edge.litertlm.Tool
import com.google.ai.edge.litertlm.ToolParam

class SampleToolSet {
    @Tool(description = "Get the current weather for a city")
    fun getCurrentWeather(
        @ToolParam(description = "The city name, e.g., San Francisco") city: String,
        @ToolParam(description = "Optional country code, e.g., US") country: String? = null,
        @ToolParam(description = "Temperature unit (celsius or fahrenheit). Default: celsius") unit: String = "celsius"
    ): Map<String, Any> {
        // In a real application, you would call a weather API here
        return mapOf("temperature" to 25, "unit" to  unit, "condition" to "Sunny")
    }

    @Tool(description = "Get the sum of a list of numbers.")
    fun sum(
        @ToolParam(description = "The numbers, could be floating point.") numbers: List<Double>,
    ): Double {
        return numbers.sum()
    }
}
```

Behind the scenes, the API inspects these annotations and the function signature
to generate an OpenAPI-style schema. This schema describes the tool's
functionality, parameters (including their types and descriptions from
`@ToolParam`), and return type to the language model.

##### Parameter Types

The types for parameters annotated with `@ToolParam` can be `String`, `Int`,
`Boolean`, `Float`, `Double`, or a `List` of these types (e.g., `List<String>`).
Use nullable types (e.g., `String?`) to indicate nullable parameters. Set a
default value to indicate that the parameters is optional, and mention the
default value in the description in `@ToolParam`.

##### Return Type

The return type of your tool function can be any Kotlin type. The result will be
converted to a JSON element before being sent back to the model.

-   `List` types are converted to json array.
-   `Map` types are converted to json object.
-   Primitive types (`String`, `Number`, `Boolean`) are converted to the corresponding json primitive.
-   Other types are converted to string with the `toString()` method.

For structured data, returning `Map` or a data class that will be converted to a
json object is recommended.

#### Registering Tools

Include instances of your tool sets in the `ConversationConfig`.

```kotlin
val conversation = engine.createConversation(
    ConversationConfig(
        tools = listOf(SampleToolSet())
        // ... other configs
    )
)


// Send messages that might trigger the tool
val userMessage = Message.of("What's the weather like in London?")
conversation.sendMessageAsync(userMessage, callback)
```

The model will decide when to call the tool based on the conversation. The
results from the tool execution are automatically sent back to the model to
generate the final response.

## Error Handling

API methods can throw `LiteRtLmJniException` for errors from the native layer or
standard Kotlin exceptions like `IllegalStateException` for lifecycle issues.
Always wrap API calls in try-catch blocks. The `onError` callback in
`MessageCallback` will also report errors during asynchronous operations.
