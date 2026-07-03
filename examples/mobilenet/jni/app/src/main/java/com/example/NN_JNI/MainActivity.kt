package com.example.NN_JNI

import android.graphics.BitmapFactory
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ColorScheme
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import kotlin.concurrent.thread

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()

        val demoDir = DemoAssetHelper.prepareDemoFiles(this)
        val modelPath = DemoAssetHelper.getModelFile(this).absolutePath
        val labelsPath = DemoAssetHelper.getLabelsFile(this).absolutePath
        val missingFiles = DemoAssetHelper.missingFiles(this)
        val assetImages = DemoAssetHelper.listAssetImages(this)

        setContent {
            var isRunning by remember { mutableStateOf(false) }
            var expanded by remember { mutableStateOf(false) }
            var selectedImage by remember { mutableStateOf(assetImages.firstOrNull() ?: "") }
            var topKResults by remember { mutableStateOf<List<Pair<String, Float>>>(emptyList()) }
            var statusMessage by remember { mutableStateOf<String?>(null) }

            MaterialTheme(colorScheme = appDarkColorScheme()) {
                Scaffold(modifier = Modifier.fillMaxSize()) { innerPadding ->
                    Column(
                        modifier = Modifier
                            .fillMaxSize()
                            .padding(innerPadding)
                            .padding(16.dp)
                            .verticalScroll(rememberScrollState()),
                        verticalArrangement = Arrangement.spacedBy(12.dp),
                    ) {
                        Card(
                            colors = CardDefaults.cardColors(
                                containerColor = MaterialTheme.colorScheme.surfaceVariant,
                            ),
                            modifier = Modifier.fillMaxWidth(),
                        ) {
                            Column(
                                modifier = Modifier.padding(14.dp),
                                verticalArrangement = Arrangement.spacedBy(12.dp),
                            ) {
                                Text("MobileNet Demo", style = MaterialTheme.typography.titleLarge)
                                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                                    Text(
                                        text = if (missingFiles.contains(DemoAssetHelper.DEFAULT_MODEL_NAME)) "Model Missing" else "Model Ready",
                                        color = if (missingFiles.contains(DemoAssetHelper.DEFAULT_MODEL_NAME)) Color(0xFFFF8A80) else Color(0xFF69F0AE),
                                    )
                                    Text(
                                        text = if (missingFiles.contains(DemoAssetHelper.DEFAULT_LABELS_NAME)) "Labels Missing" else "Labels Ready",
                                        color = if (missingFiles.contains(DemoAssetHelper.DEFAULT_LABELS_NAME)) Color(0xFFFF8A80) else Color(0xFF69F0AE),
                                    )
                                }

                                ImageSelector(
                                    selectedImage = selectedImage,
                                    imageCandidates = assetImages,
                                    expanded = expanded,
                                    onExpandedChange = { expanded = it },
                                    onSelected = {
                                        selectedImage = it
                                        expanded = false
                                        topKResults = emptyList()
                                        statusMessage = null
                                    },
                                )

                                SelectedImagePreview(selectedImage = selectedImage)

                                Button(
                                    onClick = {
                                        if (isRunning) return@Button
                                        if (missingFiles.isNotEmpty()) {
                                            topKResults = emptyList()
                                            statusMessage = "Please prepare model and labels: ${demoDir.absolutePath}"
                                            return@Button
                                        }
                                        if (selectedImage.isBlank()) {
                                            topKResults = emptyList()
                                            statusMessage = "No available test images in assets/demo"
                                            return@Button
                                        }

                                        isRunning = true
                                        topKResults = emptyList()
                                        statusMessage = null
                                        thread(name = "MobilenetDemo") {
                                            val imagePath = try {
                                                DemoAssetHelper.prepareSelectedImage(this@MainActivity, selectedImage).absolutePath
                                            } catch (e: Exception) {
                                                runOnUiThread {
                                                    isRunning = false
                                                    statusMessage = e.message ?: "Image preparation failed"
                                                }
                                                return@thread
                                            }

                                            val demo = MobilenetDemo(modelPath, imagePath, labelsPath)
                                            val result = demo.run()
                                            runOnUiThread {
                                                isRunning = false
                                                if (result.success) {
                                                    topKResults = result.topK
                                                    statusMessage = null
                                                } else {
                                                    topKResults = emptyList()
                                                    statusMessage = result.error ?: "Detection failed"
                                                }
                                            }
                                        }
                                    },
                                    enabled = !isRunning,
                                    modifier = Modifier.fillMaxWidth(),
                                    colors = ButtonDefaults.buttonColors(
                                        containerColor = Color(0xFF6750A4),
                                    ),
                                ) {
                                    Text(if (isRunning) "Detecting..." else "Start Detecting")
                                }
                            }
                        }

                        if (isRunning) {
                            Row(
                                modifier = Modifier.fillMaxWidth(),
                                horizontalArrangement = Arrangement.Center,
                                verticalAlignment = Alignment.CenterVertically,
                            ) {
                                CircularProgressIndicator(
                                    modifier = Modifier.padding(end = 12.dp),
                                    color = MaterialTheme.colorScheme.primary,
                                )
                                Text("Inferring...", color = MaterialTheme.colorScheme.onSurface)
                            }
                        }

                        statusMessage?.let { message ->
                            Text(
                                text = message,
                                color = Color(0xFFFF8A80),
                                style = MaterialTheme.typography.bodyMedium,
                            )
                        }

                        if (topKResults.isNotEmpty()) {
                            TopKResultsCard(topKResults)
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun ImageSelector(
    selectedImage: String,
    imageCandidates: List<String>,
    expanded: Boolean,
    onExpandedChange: (Boolean) -> Unit,
    onSelected: (String) -> Unit,
) {
    Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
        Text("Test Images", style = MaterialTheme.typography.labelLarge)
        OutlinedButton(
            onClick = { onExpandedChange(true) },
            enabled = imageCandidates.isNotEmpty(),
            modifier = Modifier.fillMaxWidth(),
            colors = ButtonDefaults.outlinedButtonColors(
                containerColor = Color(0xFF3A4050),
                contentColor = Color(0xFFF2F0F4),
            ),
            border = ButtonDefaults.outlinedButtonBorder.copy(
                brush = androidx.compose.ui.graphics.SolidColor(Color(0xFF8B93A7)),
            ),
        ) {
            Text(
                text = if (selectedImage.isBlank()) "Select Image" else selectedImage,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis,
            )
        }
        DropdownMenu(
            expanded = expanded,
            onDismissRequest = { onExpandedChange(false) },
            modifier = Modifier.fillMaxWidth(0.92f),
        ) {
            imageCandidates.forEach { name ->
                DropdownMenuItem(
                    text = { Text(name) },
                    onClick = { onSelected(name) },
                )
            }
        }
    }
}

@Composable
private fun SelectedImagePreview(selectedImage: String) {
    val context = LocalContext.current
    val bitmap = remember(selectedImage) {
        if (selectedImage.isBlank()) {
            null
        } else {
            runCatching {
                val file = DemoAssetHelper.prepareSelectedImage(context, selectedImage)
                BitmapFactory.decodeFile(file.absolutePath)
            }.getOrNull()
        }
    }

    Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
        Text("Image Preview", style = MaterialTheme.typography.labelLarge)
        Card(
            modifier = Modifier.fillMaxWidth(),
            colors = CardDefaults.cardColors(containerColor = Color(0xFF1A1D24)),
        ) {
            if (bitmap != null) {
                val density = LocalDensity.current
                val imageWidth = with(density) { bitmap.width.toDp() }
                val imageHeight = with(density) { bitmap.height.toDp() }
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(12.dp),
                    contentAlignment = Alignment.Center,
                ) {
                    Image(
                        bitmap = bitmap.asImageBitmap(),
                        contentDescription = selectedImage,
                        modifier = Modifier
                            .size(imageWidth, imageHeight)
                            .clip(RoundedCornerShape(4.dp)),
                        contentScale = ContentScale.None,
                    )
                }
            } else {
                Text(
                    text = "No Preview",
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(vertical = 72.dp),
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
        }
    }
}

@Composable
private fun TopKResultsCard(topK: List<Pair<String, Float>>) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = Color(0xFF1F232C)),
    ) {
        Column(
            modifier = Modifier.padding(14.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            Text(
                text = "Top-${topK.size} Classification Results",
                style = MaterialTheme.typography.titleMedium,
                color = MaterialTheme.colorScheme.onSurface,
            )
            topK.forEachIndexed { index, (label, score) ->
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Text(
                        text = "${index + 1}. $label",
                        color = MaterialTheme.colorScheme.onSurface,
                        modifier = Modifier.weight(1f),
                    )
                    Text(
                        text = "%.4f".format(score),
                        color = MaterialTheme.colorScheme.secondary,
                    )
                }
            }
        }
    }
}

private fun appDarkColorScheme(): ColorScheme = darkColorScheme(
    primary = Color(0xFFB69DF8),
    onPrimary = Color(0xFF2B1A5C),
    secondary = Color(0xFF9CCBFF),
    background = Color(0xFF121316),
    surface = Color(0xFF1B1E24),
    surfaceVariant = Color(0xFF242833),
    onSurface = Color(0xFFE6E1E5),
    onSurfaceVariant = Color(0xFFC9C5D0),
)
