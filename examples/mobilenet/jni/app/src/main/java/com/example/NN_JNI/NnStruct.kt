package com.example.NN_JNI

import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.charset.StandardCharsets

data class TensorAttr(
    val index: Int,
    val nDims: Int,
    val dims: IntArray,
    val name: String,
    val nElems: Int,
    val size: Int,
    val sizeWithStride: Int,
    val format: Int,
    val type: Int,
    val zp: Int,
    val scale: Float,
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false
        other as TensorAttr
        return index == other.index &&
            nDims == other.nDims &&
            dims.contentEquals(other.dims) &&
            name == other.name &&
            nElems == other.nElems &&
            size == other.size &&
            sizeWithStride == other.sizeWithStride &&
            format == other.format &&
            type == other.type &&
            zp == other.zp &&
            scale == other.scale
    }

    override fun hashCode(): Int {
        var result = index
        result = 31 * result + nDims
        result = 31 * result + dims.contentHashCode()
        result = 31 * result + name.hashCode()
        result = 31 * result + nElems
        result = 31 * result + size
        result = 31 * result + sizeWithStride
        result = 31 * result + format
        result = 31 * result + type
        result = 31 * result + zp
        result = 31 * result + scale.hashCode()
        return result
    }
}

data class InOutNum(
    val nInput: Int,
    val nOutput: Int,
)

object NnStruct {
    private const val MAX_DIMS = 6
    private const val MAX_NAME_LEN = 256

    fun createInOutNumBuffer(): ByteBuffer =
        NnSdk2.allocateNativeBuffer(NnSdk2.sizeOf(NnSdk2.STRUCT_IN_OUT_NUM))

    fun createTensorAttrBuffer(index: Int = 0): ByteBuffer {
        val buffer = NnSdk2.allocateNativeBuffer(NnSdk2.sizeOf(NnSdk2.STRUCT_TENSOR_ATTR))
        buffer.order(ByteOrder.nativeOrder())
        buffer.putInt(0, index)
        return buffer
    }

    fun createPerfRunBuffer(): ByteBuffer =
        NnSdk2.allocateNativeBuffer(NnSdk2.sizeOf(NnSdk2.STRUCT_PERF_RUN))

    fun readInOutNum(buffer: ByteBuffer): InOutNum {
        buffer.order(ByteOrder.nativeOrder())
        buffer.rewind()
        return InOutNum(
            nInput = buffer.int,
            nOutput = buffer.int,
        )
    }

    fun readTensorAttr(buffer: ByteBuffer): TensorAttr {
        buffer.order(ByteOrder.nativeOrder())
        buffer.rewind()

        val index = buffer.int
        val nDims = buffer.int
        val dims = IntArray(MAX_DIMS) { buffer.int }
        val nameBytes = ByteArray(MAX_NAME_LEN)
        buffer.get(nameBytes)
        val nameEnd = nameBytes.indexOf(0).let { if (it < 0) MAX_NAME_LEN else it }
        val name = String(nameBytes, 0, nameEnd, StandardCharsets.UTF_8)

        return TensorAttr(
            index = index,
            nDims = nDims,
            dims = dims,
            name = name,
            nElems = buffer.int,
            size = buffer.int,
            sizeWithStride = buffer.int,
            format = buffer.int,
            type = buffer.int,
            zp = buffer.int,
            scale = buffer.float,
        )
    }

    fun readPerfRun(buffer: ByteBuffer): Long {
        buffer.order(ByteOrder.nativeOrder())
        buffer.rewind()
        return buffer.long
    }

    fun formatTensorAttr(tag: String, attr: TensorAttr): String =
        buildString {
            append(tag)
            append('[')
            append(attr.index)
            append("] name=")
            append(attr.name)
            append(", type=")
            append(attr.type)
            append(", fmt=")
            append(attr.format)
            append(", n_elems=")
            append(attr.nElems)
            append(", size=")
            append(attr.size)
            append(", size_with_stride=")
            append(attr.sizeWithStride)
            append(", zp=")
            append(attr.zp)
            append(", scale=")
            append(attr.scale)
        }
}
