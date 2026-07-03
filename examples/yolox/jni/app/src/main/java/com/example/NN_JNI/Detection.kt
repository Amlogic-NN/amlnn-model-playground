package com.example.NN_JNI

data class Detection(
    val x1: Float,
    val y1: Float,
    val x2: Float,
    val y2: Float,
    val score: Float,
    val classId: Int,
) {
    val label: String
        get() = CocoLabels.labelOf(classId)
}

data class PreprocessMeta(
    val scale: Float,
    val padLeft: Int,
    val padTop: Int,
)

data class PreprocessResult(
    val buffer: java.nio.ByteBuffer,
    val meta: PreprocessMeta,
)
