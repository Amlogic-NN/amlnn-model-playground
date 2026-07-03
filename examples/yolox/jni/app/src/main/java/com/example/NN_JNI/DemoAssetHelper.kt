package com.example.NN_JNI

import android.content.Context
import java.io.File

object DemoAssetHelper {
    const val DEMO_DIR_NAME = "demo"
    const val DEFAULT_MODEL_NAME = "yolox_s_int8.adla"
    private val IMAGE_EXTENSIONS = setOf("jpg", "jpeg", "png", "bmp", "webp")

    fun getDemoDir(context: Context): File =
        File(context.getExternalFilesDir(null), DEMO_DIR_NAME).apply { mkdirs() }

    fun getModelFile(context: Context, fileName: String = DEFAULT_MODEL_NAME): File =
        File(getDemoDir(context), fileName)

    fun getImageFile(context: Context, fileName: String): File =
        File(getDemoDir(context), fileName)

    fun copyAssetIfExists(context: Context, assetPath: String, target: File): Boolean {
        return try {
            context.assets.open(assetPath).use { input ->
                target.parentFile?.mkdirs()
                target.outputStream().use { output -> input.copyTo(output) }
            }
            true
        } catch (_: Exception) {
            false
        }
    }

    fun prepareDemoFiles(context: Context): File {
        val demoDir = getDemoDir(context)
        val assetPrefix = "$DEMO_DIR_NAME/"
        copyAssetIfExists(context, assetPrefix + DEFAULT_MODEL_NAME, getModelFile(context))
        listAssetImages(context).forEach { imageName ->
            copyAssetIfExists(context, assetPrefix + imageName, getImageFile(context, imageName))
        }
        return demoDir
    }

    fun listAssetImages(context: Context): List<String> {
        val assetPrefix = DEMO_DIR_NAME
        return try {
            context.assets.list(assetPrefix)
                ?.filter { name ->
                    val ext = name.substringAfterLast('.', "").lowercase()
                    ext in IMAGE_EXTENSIONS
                }
                ?.sorted()
                .orEmpty()
        } catch (_: Exception) {
            emptyList()
        }
    }

    fun prepareSelectedImage(context: Context, imageName: String): File {
        val target = getImageFile(context, imageName)
        val copied = copyAssetIfExists(context, "$DEMO_DIR_NAME/$imageName", target)
        if (!copied && !target.exists()) {
            throw IllegalArgumentException("Image not found in assets/demo or demo dir: $imageName")
        }
        return target
    }

    fun missingFiles(context: Context): List<String> {
        val missing = mutableListOf<String>()
        if (!getModelFile(context).exists()) missing += DEFAULT_MODEL_NAME
        return missing
    }
}
