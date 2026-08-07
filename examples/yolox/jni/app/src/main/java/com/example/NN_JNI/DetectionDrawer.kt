package com.example.NN_JNI

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect

object DetectionDrawer {
    fun drawDetections(source: Bitmap, detections: List<Detection>): Bitmap {
        val result = source.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(result)
        val boxPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.STROKE
            strokeWidth = 3f
        }
        val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            textSize = 36f
            style = Paint.Style.FILL
        }
        val textBgPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.FILL
        }

        for (detection in detections) {
            val color = classColor(detection.classId)
            boxPaint.color = color
            textBgPaint.color = color
            textPaint.color = if (brightness(color) < 128) Color.WHITE else Color.BLACK

            val left = detection.x1.coerceIn(0f, result.width.toFloat())
            val top = detection.y1.coerceIn(0f, result.height.toFloat())
            val right = detection.x2.coerceIn(0f, result.width.toFloat())
            val bottom = detection.y2.coerceIn(0f, result.height.toFloat())

            canvas.drawRect(left, top, right, bottom, boxPaint)

            val label = "${detection.label}: ${"%.2f".format(detection.score)}"
            val textBounds = Rect()
            textPaint.getTextBounds(label, 0, label.length, textBounds)
            var labelX = left.toInt()
            var labelY = top.toInt() - 5
            if (labelY < textBounds.height()) {
                labelY = top.toInt() + textBounds.height() + 5
            }

            canvas.drawRect(
                labelX.toFloat(),
                (labelY - textBounds.height()).toFloat(),
                (labelX + textBounds.width()).toFloat(),
                (labelY + textBounds.height() / 4f),
                textBgPaint,
            )
            canvas.drawText(label, labelX.toFloat(), labelY.toFloat(), textPaint)
        }

        return result
    }

    private fun classColor(classId: Int): Int {
        val hue = (classId * 137.508f) % 360f
        val hsv = floatArrayOf(hue, 0.8f, 0.9f)
        return Color.HSVToColor(hsv)
    }

    private fun brightness(color: Int): Int {
        val r = Color.red(color)
        val g = Color.green(color)
        val b = Color.blue(color)
        return (r + g + b) / 3
    }
}
