package io.github.taehunkim.ondolsaem.engine

import android.graphics.Bitmap
import com.google.ai.edge.litert.Environment
import io.github.taehunkim.ondolsaem.engine.iris.HostOps
import io.github.taehunkim.ondolsaem.engine.iris.IrisPipeline
import io.github.taehunkim.ondolsaem.engine.iris.LiteRtExec
import io.github.taehunkim.ondolsaem.engine.iris.PromptTokens
import io.github.taehunkim.ondolsaem.engine.iris.RunFiles
import io.github.taehunkim.ondolsaem.engine.iris.StageLog
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.ByteArrayOutputStream
import java.io.File
import java.util.Random

interface ImageGenEngine : AutoCloseable {
    val modelId: String
    val backend: String get() = "GPU"
    suspend fun generateImage(prompt: String, seed: Long): ByteArray
    override fun close() {}
}

class IrisEngine(
    override val modelId: String,
    private val dir: File,
) : ImageGenEngine {
    private val runFiles by lazy {
        val rf = RunFiles(dir)
        rf.materializeTokenizer()
        rf
    }

    override suspend fun generateImage(prompt: String, seed: Long): ByteArray = withContext(Dispatchers.Default) {
        val rf = runFiles
        val tokens = PromptTokens(File(dir, "tok"), rf.embedFile())
        val env = Environment.create()
        try {
            val exec = LiteRtExec(env, dir)
            val pipeline = IrisPipeline(exec, rf.engineConfig(), rf.host, tokens, availableBytes = ::memAvailableBytes)
            val random = Random(if (seed != 0L) seed else System.currentTimeMillis())
            val noise = FloatArray(HostOps.IMAGE_FLOATS) { random.nextGaussian().toFloat() }
            val log = StageLog(clock = System::nanoTime, memory = { emptyMap() })

            val result = pipeline.generate(prompt, noise, log)

            val size = HostOps.IMAGE_SIZE
            val plane = size * size
            fun level(v: Float) = ((v + 1f) * 127.5f + 0.5f).toInt().coerceIn(0, 255)
            val pixels = IntArray(plane) { i ->
                (0xFF shl 24) or (level(result.image[i]) shl 16) or
                    (level(result.image[plane + i]) shl 8) or level(result.image[2 * plane + i])
            }
            val bitmap = Bitmap.createBitmap(pixels, size, size, Bitmap.Config.ARGB_8888)
            val stream = ByteArrayOutputStream()
            bitmap.compress(Bitmap.CompressFormat.PNG, 100, stream)
            bitmap.recycle()
            stream.toByteArray()
        } finally {
            tokens.close()
            runCatching { env.close() }
        }
    }

    companion object {
        fun load(id: String, dir: File): IrisEngine = IrisEngine(id, dir)

        private fun memAvailableBytes(): Long = runCatching {
            File("/proc/meminfo").useLines { lines ->
                for (line in lines) {
                    if (line.startsWith("MemAvailable:")) {
                        val kb = line.removePrefix("MemAvailable:").trim().substringBefore(' ').toLongOrNull()
                        if (kb != null) return@useLines kb * 1024L
                    }
                }
                Long.MAX_VALUE
            }
        }.getOrDefault(Long.MAX_VALUE)
    }
}
