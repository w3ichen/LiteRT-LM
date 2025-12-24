package com.google.ai.edge.gemmalivecall

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.widget.Button
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    private lateinit var ttsManager: TTSManager
    private lateinit var modelClient: GemmaModelClient
    private lateinit var mediaManager: MediaCaptureManager
    
    private lateinit var tvStatus: TextView
    private lateinit var tvModelOutput: TextView
    private lateinit var btnControl: Button
    
    private var isCallActive = false
    private var callJob: Job? = null
    
    // MODEL PATH (Hardcoded for demo, push file to /data/local/tmp/)
    // In a real app, copy asset to internal storage
    private val MODEL_PATH = "/data/local/tmp/gemma-3n-E2B-it-int4.litertlm"

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        tvStatus = findViewById(R.id.tvStatus)
        tvModelOutput = findViewById(R.id.tvModelOutput)
        btnControl = findViewById(R.id.btnControl)
        val viewFinder = findViewById<PreviewView>(R.id.viewFinder)

        ttsManager = TTSManager(this)
        modelClient = GemmaModelClient(this)
        mediaManager = MediaCaptureManager(this)

        if (allPermissionsGranted()) {
            startCamera(viewFinder)
        } else {
            ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS)
        }

        btnControl.setOnClickListener {
            if (isCallActive) {
                stopCall()
            } else {
                startCall()
            }
        }
        
        // Initialize Model on startup
        updateStatus("Initializing Model...")
        btnControl.isEnabled = false
        lifecycleScope.launch {
            modelClient.initialize(MODEL_PATH, 
                onSuccess = {
                    updateStatus("Model Ready")
                    btnControl.isEnabled = true
                },
                onError = { error ->
                    updateStatus("Error: $error")
                }
            )
        }
    }

    private fun startCall() {
        isCallActive = true
        btnControl.text = "Stop Call"
        updateStatus("Call Active")
        
        callJob = lifecycleScope.launch(Dispatchers.Default) {
            while (isActive && isCallActive) {
                // 1. Capture State
                withContext(Dispatchers.Main) { updateStatus("Listening...") }
                val audioBytes = mediaManager.captureAudioClip(3000) // 3 seconds
                val imageBytes = mediaManager.captureVideoFrame()
                
                if (audioBytes == null || imageBytes == null) {
                    withContext(Dispatchers.Main) { updateStatus("Capture Failed - Retrying") }
                    delay(1000)
                    continue
                }

                // 2. Inference
                withContext(Dispatchers.Main) { updateStatus("Thinking...") }
                val response = modelClient.sendAVMessage(imageBytes, audioBytes)
                
                // 3. Output
                withContext(Dispatchers.Main) {
                    tvModelOutput.text = response
                    updateStatus("Speaking...")
                    ttsManager.speak(response)
                }
                
                // Wait a bit or wait for TTS to finish (rudimentary delay here)
                delay(response.length * 100L + 1000) 
            }
        }
    }

    private fun stopCall() {
        isCallActive = false
        callJob?.cancel()
        btnControl.text = "Start Call"
        updateStatus("Call Ended")
        ttsManager.shutdown()
        // Re-init TTS as shutdown kills it
        ttsManager = TTSManager(this) 
    }

    private fun startCamera(viewFinder: PreviewView) {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()
            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(viewFinder.surfaceProvider)
            }
            
            val imageAnalyzer = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also {
                    it.setAnalyzer(Executors.newSingleThreadExecutor(), mediaManager.getImageAnalyzer())
                }

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, preview, imageAnalyzer)
            } catch (exc: Exception) {
                // Handle errors
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun updateStatus(text: String) {
        tvStatus.text = "Status: $text"
    }

    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (allPermissionsGranted()) {
                startCamera(findViewById(R.id.viewFinder))
            } else {
                // Handle permission denied
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        modelClient.close()
        ttsManager.shutdown()
    }

    companion object {
        private const val REQUEST_CODE_PERMISSIONS = 10
        private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA, Manifest.permission.RECORD_AUDIO)
    }
}
