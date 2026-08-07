package com.example.NN_JNI

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import kotlin.math.min
import kotlin.math.roundToInt

object ImagePreprocessor {
    const val MODEL_INPUT_WIDTH = 640
    const val MODEL_INPUT_HEIGHT = 640
    private const val PAD_VALUE = 114

    fun preprocessLetterbox(
        source: Bitmap,
        inputScale: Float,
        inputZeroPoint: Int,
        tensorType: Int = NnSdk2.TENSOR_INT8,
    ): PreprocessResult {
        if (tensorType != NnSdk2.TENSOR_FLOAT32) {
            require(inputScale > 0f) { "inputScale must be > 0" }
        }

        val scale = min(
            MODEL_INPUT_HEIGHT.toFloat() / source.height,
            MODEL_INPUT_WIDTH.toFloat() / source.width,
        )
        val newWidth = (source.width * scale).roundToInt()
        val newHeight = (source.height * scale).roundToInt()

        val resized = Bitmap.createScaledBitmap(source, newWidth, newHeight, true)
        val padLeft = ((MODEL_INPUT_WIDTH - newWidth) / 2.0 - 0.1).roundToInt()
        val padTop = ((MODEL_INPUT_HEIGHT - newHeight) / 2.0 - 0.1).roundToInt()

        val padded = Bitmap.createBitmap(MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(padded)
        canvas.drawColor(Color.rgb(PAD_VALUE, PAD_VALUE, PAD_VALUE))
        canvas.drawBitmap(resized, padLeft.toFloat(), padTop.toFloat(), null)
        if (resized !== source) {
            resized.recycle()
        }

        val pixels = IntArray(MODEL_INPUT_WIDTH * MODEL_INPUT_HEIGHT)
        padded.getPixels(pixels, 0, MODEL_INPUT_WIDTH, 0, 0, MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT)
        padded.recycle()

        val bytesPerPixel = if (tensorType == NnSdk2.TENSOR_FLOAT32) Float.SIZE_BYTES else 1
        val output = NnSdk2.allocateNativeBuffer(MODEL_INPUT_WIDTH * MODEL_INPUT_HEIGHT * 3 * bytesPerPixel)
        for (pixel in pixels) {
            val r = ((pixel shr 16) and 0xFF).toFloat()
            val g = ((pixel shr 8) and 0xFF).toFloat()
            val b = (pixel and 0xFF).toFloat()
            when (tensorType) {
                NnSdk2.TENSOR_FLOAT32 -> {
                    output.putFloat(r / 255f)
                    output.putFloat(g / 255f)
                    output.putFloat(b / 255f)
                }
                else -> for (channel in floatArrayOf(r, g, b)) {
                    var q = (channel / inputScale + inputZeroPoint).roundToInt()
                    q = when (tensorType) {
                        NnSdk2.TENSOR_UINT8 -> q.coerceIn(0, 255)
                        else -> q.coerceIn(-128, 127)
                    }
                    output.put(q.toByte())
                }
            }
        }
        output.rewind()

        return PreprocessResult(
            buffer = output,
            meta = PreprocessMeta(scale = scale, padLeft = padLeft, padTop = padTop),
        )
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
