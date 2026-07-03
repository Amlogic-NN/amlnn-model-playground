package com.example.NN_JNI

import android.graphics.Bitmap
import java.nio.ByteBuffer
import kotlin.math.roundToInt

object ImagePreprocessor {
    const val MODEL_INPUT_WIDTH = 224
    const val MODEL_INPUT_HEIGHT = 224

    fun preprocessQuant(
        source: Bitmap,
        inputScale: Float,
        inputZeroPoint: Int,
        tensorType: Int = NnSdk2.TENSOR_INT8,
    ): ByteBuffer {
        require(inputScale > 0f) { "inputScale must be > 0" }
        val resized = Bitmap.createScaledBitmap(
            source,
            MODEL_INPUT_WIDTH,
            MODEL_INPUT_HEIGHT,
            true,
        )

        val pixels = IntArray(MODEL_INPUT_WIDTH * MODEL_INPUT_HEIGHT)
        resized.getPixels(pixels, 0, MODEL_INPUT_WIDTH, 0, 0, MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT)
        if (resized !== source) {
            resized.recycle()
        }

        val output = NnSdk2.allocateNativeBuffer(MODEL_INPUT_WIDTH * MODEL_INPUT_HEIGHT * 3)
        var offset = 0
        for (pixel in pixels) {
            val r = (pixel shr 16) and 0xFF
            val g = (pixel shr 8) and 0xFF
            val b = pixel and 0xFF
            for (channel in intArrayOf(r, g, b)) {
                val normalized = channel / 127.5f - 1.0f
                var q = (normalized / inputScale + inputZeroPoint).roundToInt()
                q = when (tensorType) {
                    NnSdk2.TENSOR_UINT8 -> q.coerceIn(0, 255)
                    else -> q.coerceIn(-128, 127)
                }
                output.put(offset++, q.toByte())
            }
        }
        output.rewind()
        return output
    }

    fun loadBitmapFromFile(path: String, maxSide: Int = 4096): Bitmap {
        val bounds = android.graphics.BitmapFactory.Options().apply { inJustDecodeBounds = true }
        android.graphics.BitmapFactory.decodeFile(path, bounds)
        val sampleSize = calculateSampleSize(bounds.outWidth, bounds.outHeight, maxSide)
        val options = android.graphics.BitmapFactory.Options().apply { inSampleSize = sampleSize }
        return android.graphics.BitmapFactory.decodeFile(path, options)
            ?: throw IllegalArgumentException("Failed to decode image: $path")
    }

    private fun calculateSampleSize(width: Int, height: Int, maxSide: Int): Int {
        var sampleSize = 1
        var halfWidth = width / 2
        var halfHeight = height / 2
        while (halfWidth / sampleSize >= maxSide || halfHeight / sampleSize >= maxSide) {
            sampleSize *= 2
        }
        return sampleSize
    }
}
