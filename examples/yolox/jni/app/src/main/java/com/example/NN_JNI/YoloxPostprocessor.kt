package com.example.NN_JNI

import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min

object YoloxPostprocessor {
    private const val NUM_CLASSES = 80
    private const val NUM_CHANNELS = 85
    private const val OBJ_CONF_THRESHOLD = 0.1f

    fun getTensorShape(attr: TensorAttr): IntArray =
        attr.dims.filter { it > 1 }.toIntArray()

    fun postprocess(
        output: FloatArray,
        outputShape: IntArray,
        meta: PreprocessMeta,
        confThreshold: Float,
        iouThreshold: Float,
    ): List<Detection> {
        if (outputShape.isEmpty()) return emptyList()

        val channelsLast = outputShape.last() == NUM_CHANNELS
        val numAnchors = if (channelsLast) {
            outputShape[outputShape.size - 2]
        } else {
            outputShape.last()
        }

        val detections = ArrayList<Detection>()
        for (anchor in 0 until numAnchors) {
            val objConf = output[indexOf(channelsLast, anchor, 4, numAnchors)]
            if (objConf < OBJ_CONF_THRESHOLD) continue

            var maxClsScore = 0f
            var classId = -1
            for (classIndex in 0 until NUM_CLASSES) {
                val clsScore = output[indexOf(channelsLast, anchor, 5 + classIndex, numAnchors)]
                if (clsScore > maxClsScore) {
                    maxClsScore = clsScore
                    classId = classIndex
                }
            }

            val finalScore = objConf * maxClsScore
            if (finalScore < confThreshold || classId < 0) continue

            val (gridX, gridY, stride) = gridInfo(anchor)
            val rawCx = output[indexOf(channelsLast, anchor, 0, numAnchors)]
            val rawCy = output[indexOf(channelsLast, anchor, 1, numAnchors)]
            val rawW = output[indexOf(channelsLast, anchor, 2, numAnchors)]
            val rawH = output[indexOf(channelsLast, anchor, 3, numAnchors)]

            val cx = (rawCx + gridX) * stride
            val cy = (rawCy + gridY) * stride
            val w = exp(rawW.toDouble()).toFloat() * stride
            val h = exp(rawH.toDouble()).toFloat() * stride

            val x1 = cx - w / 2f
            val y1 = cy - h / 2f
            val x2 = cx + w / 2f
            val y2 = cy + h / 2f

            detections.add(
                Detection(
                    x1 = max(0f, (x1 - meta.padLeft) / meta.scale),
                    y1 = max(0f, (y1 - meta.padTop) / meta.scale),
                    x2 = max(0f, (x2 - meta.padLeft) / meta.scale),
                    y2 = max(0f, (y2 - meta.padTop) / meta.scale),
                    score = finalScore,
                    classId = classId,
                ),
            )
        }

        return nmsByClass(detections, iouThreshold)
    }

    private fun indexOf(channelsLast: Boolean, anchor: Int, channel: Int, numAnchors: Int): Int =
        if (channelsLast) anchor * NUM_CHANNELS + channel else channel * numAnchors + anchor

    private fun gridInfo(anchor: Int): Triple<Int, Int, Int> =
        when {
            anchor < 6400 -> Triple(anchor % 80, anchor / 80, 8)
            anchor < 8000 -> {
                val local = anchor - 6400
                Triple(local % 40, local / 40, 16)
            }
            else -> {
                val local = anchor - 8000
                Triple(local % 20, local / 20, 32)
            }
        }

    private fun nmsByClass(detections: List<Detection>, iouThreshold: Float): List<Detection> {
        if (detections.isEmpty()) return emptyList()

        val finalDetections = ArrayList<Detection>()
        detections.groupBy { it.classId }.values.forEach { classDets ->
            val sorted = classDets.sortedByDescending { it.score }
            val removed = BooleanArray(sorted.size)
            for (i in sorted.indices) {
                if (removed[i]) continue
                finalDetections.add(sorted[i])
                for (j in i + 1 until sorted.size) {
                    if (removed[j]) continue
                    if (computeIou(sorted[i], sorted[j]) > iouThreshold) {
                        removed[j] = true
                    }
                }
            }
        }
        return finalDetections.sortedByDescending { it.score }
    }

    private fun computeIou(a: Detection, b: Detection): Float {
        val xx1 = max(a.x1, b.x1)
        val yy1 = max(a.y1, b.y1)
        val xx2 = min(a.x2, b.x2)
        val yy2 = min(a.y2, b.y2)

        val w = max(0f, xx2 - xx1)
        val h = max(0f, yy2 - yy1)
        val inter = w * h

        val area1 = (a.x2 - a.x1) * (a.y2 - a.y1)
        val area2 = (b.x2 - b.x1) * (b.y2 - b.y1)
        return inter / (area1 + area2 - inter)
    }
}
