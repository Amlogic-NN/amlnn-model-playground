package com.example.NN_JNI

import android.graphics.Bitmap
import android.util.Log
import java.io.File
import kotlin.math.exp

data class ClassScore(
    val index: Int,
    val score: Float,
)

data class DemoTiming(
    var preprocessMs: Double = 0.0,
    var initMs: Double = 0.0,
    var inputSetMs: Double = 0.0,
    var runMs: Double = 0.0,
    var outputGetMs: Double = 0.0,
    var releaseMs: Double = 0.0,
    var totalMs: Double = 0.0,
)

data class DemoResult(
    val success: Boolean,
    val topK: List<Pair<String, Float>>,
    val timing: DemoTiming,
    val logText: String,
    val error: String? = null,
)

class MobilenetDemo(
    private val modelPath: String,
    private val imagePath: String,
    private val labelsPath: String,
) {
    companion object {
        private const val TAG = "MobilenetDemo"
        private const val TOP_K = 5
        private const val INIT_TIMEOUT_MS = 120_000
    }

    fun run(): DemoResult {
        val log = StringBuilder()
        val timing = DemoTiming()
        var contextHandle = 0L

        fun append(line: String) {
            log.appendLine(line)
            Log.i(TAG, line)
        }

        return try {
            val totalStart = System.nanoTime()
            append("MobileNet Quant Demo (AMLNN SDK2 - Android JNI)")
            append("Model: $modelPath")
            append("Image: $imagePath")
            append("Labels: $labelsPath")

            if (!File(modelPath).exists()) error("Model file not found: $modelPath")
            if (!File(imagePath).exists()) error("Image file not found: $imagePath")
            if (!File(labelsPath).exists()) error("Labels file not found: $labelsPath")

            lateinit var inAttrs: List<TensorAttr>
            lateinit var outAttrs: List<TensorAttr>
            timing.initMs = measureMs {
                contextHandle = NnSdk2.initFromFile(
                    modelPath = modelPath,
                    backendType = NnSdk2.BACKEND_ADLA_NPU,
                    taskPriority = NnSdk2.TASK_PRIORITY_MEDIUM,
                    timeoutMs = INIT_TIMEOUT_MS,
                )
                if (contextHandle == 0L) {
                    error("amlnn_init returned invalid context")
                }
                val tensorAttrs = queryTensorAttrs(contextHandle)
                inAttrs = tensorAttrs.first
                outAttrs = tensorAttrs.second
            }

            inAttrs.forEach { append(NnStruct.formatTensorAttr("Input ", it)) }
            outAttrs.forEach { append(NnStruct.formatTensorAttr("Output", it)) }

            val inAttr = inAttrs[0]
            append("Input quant params: type=${inAttr.type}, scale=${inAttr.scale}, zp=${inAttr.zp}")

            var bitmap: Bitmap? = null
            timing.preprocessMs = measureMs {
                bitmap = ImagePreprocessor.loadBitmapFromFile(imagePath)
            }

            val inputBuffer = try {
                var buffer = NnSdk2.allocateNativeBuffer(0)
                timing.preprocessMs += measureMs {
                    buffer = ImagePreprocessor.preprocessQuant(
                        source = bitmap!!,
                        inputScale = inAttr.scale,
                        inputZeroPoint = inAttr.zp,
                        tensorType = inAttr.type,
                    )
                    append(
                        bufferStatsQuant(
                            "Input quantized",
                            buffer,
                            inAttr.type,
                            ImagePreprocessor.MODEL_INPUT_WIDTH * ImagePreprocessor.MODEL_INPUT_HEIGHT * 3,
                        ),
                    )
                }
                buffer
            } finally {
                bitmap?.recycle()
            }

            val inBytes = if (inAttr.nElems > 0) {
                inAttr.nElems
            } else if (inAttr.size > 0) {
                inAttr.size
            } else {
                ImagePreprocessor.MODEL_INPUT_WIDTH * ImagePreprocessor.MODEL_INPUT_HEIGHT * 3
            }

            timing.inputSetMs = measureMs {
                checkResult(
                    NnSdk2.inputsSet(
                        context = contextHandle,
                        buffers = arrayOf(inputBuffer),
                        sizes = intArrayOf(inBytes),
                        indices = intArrayOf(0),
                    ),
                    "amlnn_inputs_set",
                )
            }

            timing.runMs = measureMs {
                checkResult(NnSdk2.run(contextHandle), "amlnn_run")
            }

            lateinit var logits: FloatArray
            timing.outputGetMs = measureMs {
                logits = NnSdk2.outputsGet(contextHandle, 0)
            }

            append(bufferStatsFloat("Output logits", logits))
            val scores = softmax(logits)
            append(bufferStatsFloat("Output softmax", scores))

            val topK = buildTopK(scores, loadLabels(labelsPath))
            append("")
            append("Top-$TOP_K Classification Results:")
            topK.forEachIndexed { rank, (label, score) ->
                append("  ${rank + 1}. $label (score: ${"%.6f".format(score)})")
            }

            timing.releaseMs = measureMs {
                NnSdk2.destroy(contextHandle)
                contextHandle = 0
            }

            timing.totalMs = (System.nanoTime() - totalStart) / 1_000_000.0

            append("")
            append(formatTimingSummary(timing))

            return DemoResult(
                success = true,
                topK = topK,
                timing = timing,
                logText = log.toString(),
            )
        } catch (e: Exception) {
            Log.e(TAG, "Demo failed", e)
            append("Error: ${e.message}")
            DemoResult(
                success = false,
                topK = emptyList(),
                timing = timing,
                logText = log.toString(),
                error = e.message,
            )
        } finally {
            if (contextHandle != 0L) {
                NnSdk2.destroy(contextHandle)
            }
        }
    }

    private fun queryTensorAttrs(
        contextHandle: Long,
    ): Pair<List<TensorAttr>, List<TensorAttr>> {
        val ioBuffer = NnStruct.createInOutNumBuffer()
        checkResult(NnSdk2.query(contextHandle, NnSdk2.QUERY_IN_OUT_NUM, ioBuffer), "QUERY_IN_OUT_NUM")
        val ioNum = NnStruct.readInOutNum(ioBuffer)

        val inAttrs = (0 until ioNum.nInput).map { index ->
            val attrBuffer = NnStruct.createTensorAttrBuffer(index)
            checkResult(
                NnSdk2.query(contextHandle, NnSdk2.QUERY_INPUT_ATTR, attrBuffer),
                "QUERY_INPUT_ATTR[$index]",
            )
            NnStruct.readTensorAttr(attrBuffer)
        }
        val outAttrs = (0 until ioNum.nOutput).map { index ->
            val attrBuffer = NnStruct.createTensorAttrBuffer(index)
            checkResult(
                NnSdk2.query(contextHandle, NnSdk2.QUERY_OUTPUT_ATTR, attrBuffer),
                "QUERY_OUTPUT_ATTR[$index]",
            )
            NnStruct.readTensorAttr(attrBuffer)
        }
        if (inAttrs.isEmpty() || outAttrs.isEmpty()) {
            error("Failed to query model tensor attributes")
        }
        return inAttrs to outAttrs
    }

    private inline fun measureMs(block: () -> Unit): Double {
        val start = System.nanoTime()
        block()
        return (System.nanoTime() - start) / 1_000_000.0
    }

    private fun formatMs(ms: Double): String = "%.3f".format(ms)

    private fun checkResult(ret: Int, tag: String) {
        if (ret != NnSdk2.AMLNN_SUCCESS) {
            error("$tag failed, ret=$ret")
        }
    }

    private fun softmax(logits: FloatArray): FloatArray {
        if (logits.isEmpty()) return logits
        val max = logits.maxOrNull() ?: 0f
        var sum = 0.0
        val values = FloatArray(logits.size) { index ->
            exp((logits[index] - max).toDouble()).also { sum += it }.toFloat()
        }
        if (sum == 0.0) return values
        for (index in values.indices) {
            values[index] = (values[index] / sum).toFloat()
        }
        return values
    }

    private fun bufferStatsQuant(tag: String, buffer: java.nio.ByteBuffer, tensorType: Int, count: Int): String {
        if (count <= 0) return "$tag: empty buffer"
        buffer.rewind()
        var minV = Int.MAX_VALUE
        var maxV = Int.MIN_VALUE
        var nonzero = 0
        var sum = 0L
        repeat(count) {
            val value = if (tensorType == NnSdk2.TENSOR_UINT8) {
                buffer.get().toInt() and 0xFF
            } else {
                buffer.get().toInt()
            }
            minV = minOf(minV, value)
            maxV = maxOf(maxV, value)
            if (value != 0) nonzero++
            sum += value
        }
        buffer.rewind()
        return "$tag: min=$minV, max=$maxV, nonzero=$nonzero/$count, mean=${sum.toDouble() / count}"
    }

    private fun bufferStatsFloat(tag: String, values: FloatArray): String {
        if (values.isEmpty()) return "$tag: empty buffer"
        var minV = Float.MAX_VALUE
        var maxV = -Float.MAX_VALUE
        var nonzero = 0
        var sum = 0.0
        for (value in values) {
            minV = minOf(minV, value)
            maxV = maxOf(maxV, value)
            if (value != 0f) nonzero++
            sum += value
        }
        return "$tag: min=$minV, max=$maxV, nonzero=$nonzero/${values.size}, mean=${sum / values.size}"
    }

    private fun loadLabels(path: String): List<String> =
        File(path).readLines().map { it.trimEnd() }

    private fun buildTopK(scores: FloatArray, labels: List<String>): List<Pair<String, Float>> =
        scores.indices
            .map { ClassScore(it, scores[it]) }
            .sortedByDescending { it.score }
            .take(TOP_K)
            .map { item ->
                val label = labels.getOrNull(item.index) ?: "Class ${item.index}"
                label to item.score
            }

    private fun formatTimingSummary(timing: DemoTiming): String =
        buildString {
            appendLine("========== [Android JNI] Timing Summary ==========")
            appendLine("  preprocess:  ${formatMs(timing.preprocessMs)} ms")
            appendLine("  init:        ${formatMs(timing.initMs)} ms")
            appendLine("  input_set:   ${formatMs(timing.inputSetMs)} ms")
            appendLine("  run:         ${formatMs(timing.runMs)} ms")
            appendLine("  output_get:  ${formatMs(timing.outputGetMs)} ms")
            appendLine("  release:     ${formatMs(timing.releaseMs)} ms")
            appendLine("  total:       ${formatMs(timing.totalMs)} ms")
            append("==================================================")
        }
}
