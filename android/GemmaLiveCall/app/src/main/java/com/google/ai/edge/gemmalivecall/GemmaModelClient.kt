package com.google.ai.edge.gemmalivecall

import android.content.Context
import android.util.Log
import com.google.ai.edge.litertlm.Backend
import com.google.ai.edge.litertlm.Content
import com.google.ai.edge.litertlm.Conversation
import com.google.ai.edge.litertlm.Engine
import com.google.ai.edge.litertlm.EngineConfig
import com.google.ai.edge.litertlm.Message
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File

class GemmaModelClient(private val context: Context) {
    private var engine: Engine? = null
    private var conversation: Conversation? = null
    private val TAG = "GemmaModelClient"

    suspend fun initialize(modelPath: String, onSuccess: () -> Unit, onError: (String) -> Unit) {
        withContext(Dispatchers.IO) {
            try {
                val file = File(modelPath)
                if (!file.exists()) {
                    withContext(Dispatchers.Main) { onError("Model file not found at $modelPath") }
                    return@withContext
                }

                // Config for Gemma-3n (Native Multimodal)
                // We use CPU for the main LLM execution to avoid GPU driver crashes/internal errors
                // on devices with limited OpenCL support.
                // However, we MUST use GPU for the Vision backend as the model requires it.
                // Limiting maxNumTokens to lower memory usage on older devices.
                val config = EngineConfig(
                    modelPath = modelPath,
                    backend = Backend.CPU, 
                    visionBackend = Backend.GPU,
                    audioBackend = Backend.CPU,
                    maxNumTokens = 256
                )
                
                Log.d(TAG, "Initializing Engine with backend: CPU (LLM), GPU (Vision), MaxTokens: 256...")
                engine = Engine(config)
                engine?.initialize()
                
                Log.d(TAG, "Creating Conversation...")
                conversation = engine?.createConversation()
                
                withContext(Dispatchers.Main) { onSuccess() }
            } catch (e: Exception) {
                Log.e(TAG, "Failed to initialize", e)
                withContext(Dispatchers.Main) { onError(e.message ?: "Unknown error") }
            }
        }
    }

    suspend fun sendAVMessage(
        imageBytes: ByteArray, 
        audioBytes: ByteArray, 
        prompt: String = "Respond to this video and audio."
    ): String {
        return withContext(Dispatchers.IO) {
            try {
                val conv = conversation ?: throw IllegalStateException("Conversation not initialized")
                
                // Construct Multimodal Message
                val message = Message.of(
                    Content.ImageBytes(imageBytes),
                    Content.AudioBytes(audioBytes),
                    Content.Text(prompt)
                )
                
                Log.d(TAG, "Sending message...")
                val response = conv.sendMessage(message)
                Log.d(TAG, "Response received: $response")
                
                response.toString()
            } catch (e: Exception) {
                Log.e(TAG, "Inference failed", e)
                "Error: ${e.message}"
            }
        }
    }

    fun close() {
        conversation?.close()
        engine?.close()
    }
}
