package com.example.NN_JNI

import android.graphics.Bitmap
import android.util.Log
import java.io.File

data class DemoResult(
    val success: Boolean,
    val detections: List<Detection>,
    val resultBitmap: Bitmap?,
    val error: String? = null,
)

class YoloxDemo(
    private val modelPath: String,
    private val imagePath: String,
) {
    companion object {
        private const val TAG = "YoloxDemo"
        private const val INIT_TIMEOUT_MS = 120_000
        private const val SCORE_THRESHOLD = 0.3f
        private const val NMS_THRESHOLD = 0.2f
    }

    fun run(): DemoResult {
        var contextHandle = 0L
        var sourceBitmap: Bitmap? = null

        fun log(line: String) = Log.i(TAG, line)

        return try {
            val totalStart = System.nanoTime()
            log("YOLOX Demo (AMLNN SDK2 - Android JNI)")
            log("Model: $modelPath")
            log("Image: $imagePath")

            if (!File(modelPath).exists()) error("Model file not found: $modelPath")
            if (!File(imagePath).exists()) error("Image file not found: $imagePath")

            lateinit var inAttrs: List<TensorAttr>
            lateinit var outAttrs: List<TensorAttr>
            val initMs = measureMs {
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
            inAttrs.forEach { log(NnStruct.formatTensorAttr("Input ", it)) }
            outAttrs.forEach { log(NnStruct.formatTensorAttr("Output", it)) }

            val inAttr = inAttrs[0]
            val outputShape = YoloxPostprocessor.getTensorShape(outAttrs[0])
            log("Output shape: ${outputShape.joinToString(prefix = "[", postfix = "]")}")

            lateinit var preprocessResult: PreprocessResult
            val preprocessMs = measureMs {
                sourceBitmap = ImagePreprocessor.loadBitmapFromFile(imagePath)
                preprocessResult = ImagePreprocessor.preprocessLetterbox(
                    source = sourceBitmap!!,
                    inputScale = inAttr.scale,
                    inputZeroPoint = inAttr.zp,
                    tensorType = inAttr.effectiveType(),
                )
            }
            val inputSetMs = measureMs {
                checkResult(
                    NnSdk2.inputsSet(
                        context = contextHandle,
                        buffers = arrayOf(preprocessResult.buffer),
                        sizes = intArrayOf(
                            inAttr.byteSize(
                                ImagePreprocessor.MODEL_INPUT_WIDTH *
                                    ImagePreprocessor.MODEL_INPUT_HEIGHT * 3,
                            ),
                        ),
                        indices = intArrayOf(0),
                    ),
                    "amlnn_inputs_set",
                )
            }
            val runMs = measureMs {
                checkResult(NnSdk2.run(contextHandle), "amlnn_run")
            }
            lateinit var output: FloatArray
            val outputGetMs = measureMs {
                output = NnSdk2.outputsGet(contextHandle, 0)
            }
            val detections = YoloxPostprocessor.postprocess(
                output = output,
                outputShape = outputShape,
                meta = preprocessResult.meta,
                confThreshold = SCORE_THRESHOLD,
                iouThreshold = NMS_THRESHOLD,
            )

            log("Detections: ${detections.size}")
            detections.forEachIndexed { index, detection ->
                log(
                    "  ${index + 1}. ${detection.label} " +
                        "(${detection.x1.toInt()},${detection.y1.toInt()})-" +
                        "(${detection.x2.toInt()},${detection.y2.toInt()}) " +
                        "score=${"%.4f".format(detection.score)}",
                )
            }

            val resultBitmap = sourceBitmap?.let { DetectionDrawer.drawDetections(it, detections) }

            val releaseMs = measureMs {
                NnSdk2.destroy(contextHandle)
                contextHandle = 0
            }
            val totalMs = (System.nanoTime() - totalStart) / 1_000_000.0
            log("")
            log(formatTimingSummary(initMs, preprocessMs, inputSetMs, runMs, outputGetMs, releaseMs, totalMs))

            DemoResult(
                success = true,
                detections = detections,
                resultBitmap = resultBitmap,
            )
        } catch (e: Exception) {
            Log.e(TAG, "Demo failed", e)
            log("Error: ${e.message}")
            DemoResult(
                success = false,
                detections = emptyList(),
                resultBitmap = null,
                error = e.message,
            )
        } finally {
            sourceBitmap?.recycle()
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

    private fun formatTimingSummary(
        initMs: Double,
        preprocessMs: Double,
        inputSetMs: Double,
        runMs: Double,
        outputGetMs: Double,
        releaseMs: Double,
        totalMs: Double,
    ): String = buildString {
        appendLine("========== [Android JNI] Timing Summary ==========")
        appendLine("  preprocess:  ${formatMs(preprocessMs)} ms")
        appendLine("  init:        ${formatMs(initMs)} ms")
        appendLine("  input_set:   ${formatMs(inputSetMs)} ms")
        appendLine("  run:         ${formatMs(runMs)} ms")
        appendLine("  output_get:  ${formatMs(outputGetMs)} ms")
        appendLine("  release:     ${formatMs(releaseMs)} ms")
        appendLine("  total:       ${formatMs(totalMs)} ms")
        append("==================================================")
    }
}
