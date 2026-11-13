package io.github.taehunkim.ondolsaem.engine

import android.content.Context
import com.k2fsa.sherpa.onnx.OfflineModelConfig
import com.k2fsa.sherpa.onnx.OfflineRecognizer
import com.k2fsa.sherpa.onnx.OfflineRecognizerConfig
import com.k2fsa.sherpa.onnx.OfflineSenseVoiceModelConfig
import com.k2fsa.sherpa.onnx.SileroVadModelConfig
import com.k2fsa.sherpa.onnx.Vad
import com.k2fsa.sherpa.onnx.VadModelConfig
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import io.github.taehunkim.ondolsaem.sdk.SpeechLanguages
import java.io.File
import java.io.RandomAccessFile

/** Voice activity detection and speech recognition on a decoded PCM file (16 kHz mono 16-bit). */
interface SpeechTools : AutoCloseable {
    val canTranscribe: Boolean

    /** Time ranges that contain speech (voice activity detection only). */
    suspend fun speechSpans(pcm: File): List<Span>

    /** Streams recognised utterances from [fromMs] on. Needs an ASR model ([canTranscribe]). */
    suspend fun transcribe(pcm: File, fromMs: Long, onCue: suspend (Cue) -> Unit)

    /** Override when an engine supports explicit language hints. Legacy/fake engines may still support auto only. */
    suspend fun transcribeWithLanguage(pcm: File, fromMs: Long, language: String, onCue: suspend (Cue) -> Unit) {
        require(language == "auto") { "This speech engine does not support explicit languages" }
        transcribe(pcm, fromMs, onCue)
    }

    override fun close() {}
}

/**
 * sherpa-onnx: Silero VAD (bundled, 0.6 MB) finds utterances, SenseVoice (downloaded) recognises them.
 * [asrDir] is null when no recognition model is installed; VAD still works then.
 */
class SherpaSpeech(private val context: Context, private val asrDir: File?, private val threads: Int = 2) : SpeechTools {
    private val recognitionMutex = Mutex()
    private var activeLanguage = "auto"
    private var recognizer: OfflineRecognizer? = createRecognizer("auto")

    private fun createRecognizer(language: String): OfflineRecognizer? = asrDir?.let {
        OfflineRecognizer(
            null,
            OfflineRecognizerConfig(
                modelConfig = OfflineModelConfig(
                    senseVoice = OfflineSenseVoiceModelConfig(model = File(it, "model.int8.onnx").path, language = language, useInverseTextNormalization = true),
                    tokens = File(it, "tokens.txt").path,
                    numThreads = threads,
                ),
            ),
        )
    }

    override val canTranscribe: Boolean get() = asrDir != null

    private fun newVad() = Vad(
        context.assets,
        VadModelConfig(
            sileroVadModelConfig = SileroVadModelConfig(
                model = VAD_ASSET, threshold = 0.5f, minSilenceDuration = 0.4f, minSpeechDuration = 0.25f,
                windowSize = 512, maxSpeechDuration = 20f,
            ),
            sampleRate = Pcm.RATE, numThreads = 1,
        ),
    )

    override suspend fun speechSpans(pcm: File): List<Span> {
        val out = ArrayList<Span>()
        runVad(pcm, 0) { startMs, samples -> out += Span(startMs, startMs + samples.size * 1000L / Pcm.RATE) }
        return out
    }

    override suspend fun transcribe(pcm: File, fromMs: Long, onCue: suspend (Cue) -> Unit) =
        transcribeWithLanguage(pcm, fromMs, "auto", onCue)

    override suspend fun transcribeWithLanguage(pcm: File, fromMs: Long, language: String, onCue: suspend (Cue) -> Unit) = recognitionMutex.withLock {
        SpeechLanguages.requireSupported(language)
        if (language != activeLanguage || recognizer == null) {
            // Keep only one model resident. The ASR service gate serializes requests; this mutex also protects direct calls.
            recognizer?.release()
            recognizer = null
            recognizer = createRecognizer(language)
            activeLanguage = language
        }
        val r = recognizer ?: return@withLock
        runVad(pcm, fromMs) { startMs, samples ->
            val stream = r.createStream()
            try {
                stream.acceptWaveform(samples, Pcm.RATE)
                r.decode(stream)
                val text = r.getResult(stream).text.trim()
                if (text.isNotEmpty()) onCue(Cue(startMs, startMs + samples.size * 1000L / Pcm.RATE, text))
            } finally { stream.release() }
        }
    }

    /** Feeds the PCM file to the VAD in small blocks and hands every detected utterance to [onSegment]. */
    private suspend fun runVad(pcm: File, fromMs: Long, onSegment: suspend (startMs: Long, samples: FloatArray) -> Unit) {
        val vad = newVad()
        try {
            RandomAccessFile(pcm, "r").use { f ->
                f.seek(Pcm.msToByte(fromMs).coerceAtMost(f.length()))
                val buf = ByteArray(BLOCK_SAMPLES * 2)
                suspend fun drain() {
                    while (!vad.empty()) {
                        currentCoroutineContext().ensureActive()
                        val seg = vad.front()
                        onSegment(fromMs + seg.start * 1000L / Pcm.RATE, seg.samples)
                        vad.pop()
                    }
                }
                while (true) {
                    val n = f.read(buf)
                    if (n <= 0) break
                    vad.acceptWaveform(Pcm.toFloats(buf.copyOf(n - n % 2)))
                    drain()
                }
                vad.flush()
                drain()
            }
        } finally { vad.release() }
    }

    override fun close() { recognizer?.release() }

    companion object {
        /** Fetched by scripts/fetch-sherpa-onnx.sh into app/src/main/assets (not committed). */
        const val VAD_ASSET = "silero_vad.onnx"
        private const val BLOCK_SAMPLES = 512
    }
}
