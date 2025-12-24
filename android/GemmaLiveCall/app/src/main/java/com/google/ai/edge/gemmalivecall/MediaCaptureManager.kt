package com.google.ai.edge.gemmalivecall

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.YuvImage
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.core.app.ActivityCompat
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer
import java.util.concurrent.atomic.AtomicReference

class MediaCaptureManager(private val context: Context) {
    
    // Holds the latest camera frame
    private val latestFrame = AtomicReference<ByteArray?>(null)

    // ImageAnalysis.Analyzer for CameraX
    fun getImageAnalyzer(): ImageAnalysis.Analyzer {
        return ImageAnalysis.Analyzer { image ->
            val bytes = imageProxyToJpeg(image)
            latestFrame.set(bytes)
            image.close()
        }
    }

    fun captureVideoFrame(): ByteArray? {
        return latestFrame.get()
    }

    fun captureAudioClip(durationMs: Int): ByteArray? {
        if (ActivityCompat.checkSelfPermission(context, Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            return null
        }

        val sampleRate = 16000 // Standard for models
        val channelConfig = AudioFormat.CHANNEL_IN_MONO
        val audioFormat = AudioFormat.ENCODING_PCM_16BIT
        val minBufSize = AudioRecord.getMinBufferSize(sampleRate, channelConfig, audioFormat)
        val bufferSize = maxOf(minBufSize, sampleRate * 2 * (durationMs / 1000)) // Enough for duration

        val recorder = AudioRecord(MediaRecorder.AudioSource.MIC, sampleRate, channelConfig, audioFormat, bufferSize)
        val buffer = ByteArray(bufferSize)

        try {
            recorder.startRecording()
            // Read for specified duration (approx)
            // Note: This is a blocking read for simplicity in this demo loop
            // In production, use a separate thread or non-blocking read
            var bytesRead = 0
            val chunks = ArrayList<ByteArray>()
            
            // Read until we have enough data (very rough approx for demo)
            // For 16kHz 16bit mono, 1 sec = 32000 bytes.
            val targetBytes = (sampleRate * 2 * (durationMs / 1000.0)).toInt()
            
            while (bytesRead < targetBytes) {
                val read = recorder.read(buffer, 0, minOf(buffer.size, targetBytes - bytesRead))
                if (read > 0) {
                    val chunk = ByteArray(read)
                    System.arraycopy(buffer, 0, chunk, 0, read)
                    chunks.add(chunk)
                    bytesRead += read
                } else {
                    break
                }
            }
            
            recorder.stop()
            
            // Combine chunks
            val out = ByteArrayOutputStream()
            for (chunk in chunks) {
                out.write(chunk)
            }
            return out.toByteArray()

        } catch (e: Exception) {
            e.printStackTrace()
            return null
        } finally {
            recorder.release()
        }
    }

    private fun imageProxyToJpeg(image: ImageProxy): ByteArray {
        val yBuffer = image.planes[0].buffer // Y
        val uBuffer = image.planes[1].buffer // U
        val vBuffer = image.planes[2].buffer // V

        val ySize = yBuffer.remaining()
        val uSize = uBuffer.remaining()
        val vSize = vBuffer.remaining()

        val nv21 = ByteArray(ySize + uSize + vSize)

        yBuffer.get(nv21, 0, ySize)
        vBuffer.get(nv21, ySize, vSize)
        uBuffer.get(nv21, ySize + vSize, uSize)

        val yuvImage = YuvImage(nv21, ImageFormat.NV21, image.width, image.height, null)
        val out = ByteArrayOutputStream()
        yuvImage.compressToJpeg(Rect(0, 0, image.width, image.height), 80, out) // 80 quality for speed
        return out.toByteArray()
    }
}
