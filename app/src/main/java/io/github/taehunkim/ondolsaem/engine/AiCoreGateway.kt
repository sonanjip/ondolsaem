package io.github.taehunkim.ondolsaem.engine

import android.util.Log
import com.google.mlkit.genai.common.DownloadStatus
import com.google.mlkit.genai.common.FeatureStatus
import com.google.mlkit.genai.common.FunctionCall
import com.google.mlkit.genai.prompt.Feature
import com.google.mlkit.genai.prompt.GenerateContentRequest
import com.google.mlkit.genai.prompt.Generation
import com.google.mlkit.genai.prompt.GenerativeModel
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.map

/** Single streamed emission from AICore Gemini Nano. */
data class NanoStreamChunk(
    val text: String,
    val functionCalls: List<FunctionCall> = emptyList(),
)

/** Session handle for running Gemini Nano inference through Android AICore. */
interface NanoChatSession : AutoCloseable {
    val supportsSystemPrompt: Boolean
    val supportsFunctionCalling: Boolean
    val tokenLimit: Int

    fun generateStream(request: GenerateContentRequest): Flow<NanoStreamChunk>
    suspend fun clearCaches()
}

/** Abstraction over ML Kit GenAI Prompt API (`Generation.getClient()`), enabling JVM unit tests. */
interface AiCoreGateway {
    suspend fun checkStatus(): Int
    fun download(): Flow<DownloadStatus>
    suspend fun openChatSession(): NanoChatSession
}

/** Default [AiCoreGateway] backed by `com.google.mlkit:genai-prompt` and Android AICore. */
class MlKitAiCoreGateway(
    private val clientFactory: () -> GenerativeModel = { Generation.getClient() },
) : AiCoreGateway {

    override suspend fun checkStatus(): Int = try {
        clientFactory().useClient { it.checkStatus() }
    } catch (e: CancellationException) {
        throw e
    } catch (t: Throwable) {
        Log.d(TAG, "AICore status check unavailable", t)
        FeatureStatus.UNAVAILABLE
    }

    override fun download(): Flow<DownloadStatus> = kotlinx.coroutines.flow.flow {
        val client = clientFactory()
        try {
            client.download().collect { emit(it) }
        } finally {
            runCatching { client.close() }
        }
    }

    override suspend fun openChatSession(): NanoChatSession {
        val client = clientFactory()
        try {
            val status = client.checkStatus()
            check(status == FeatureStatus.AVAILABLE) { "Gemini Nano is not ready in AICore (status=$status)" }
            val features = runCatching { client.getAvailableFeatures() }.getOrDefault(emptyList())
            val systemPrompt = Feature.SYSTEM_PROMPT in features
            val functionCalling = Feature.FUNCTION_CALLING in features
            val tokenLimit = runCatching { client.getTokenLimit() }.getOrNull()
                ?.takeIf { it > 0 } ?: GeminiNanoChatModel.MAX_CONTEXT_TOKENS
            runCatching { client.warmup() }
                .onFailure { Log.w(TAG, "AICore warmup failed; continuing", it) }
            Log.i(
                TAG,
                "AICore Gemini Nano ready (systemPrompt=$systemPrompt, functionCalling=$functionCalling, tokenLimit=$tokenLimit)",
            )
            return MlKitNanoChatSession(client, systemPrompt, functionCalling, tokenLimit)
        } catch (t: Throwable) {
            runCatching { client.close() }
            throw t
        }
    }

    private suspend inline fun <T> GenerativeModel.useClient(block: (GenerativeModel) -> T): T = try {
        block(this)
    } finally {
        runCatching { close() }
    }

    private class MlKitNanoChatSession(
        private val client: GenerativeModel,
        override val supportsSystemPrompt: Boolean,
        override val supportsFunctionCalling: Boolean,
        override val tokenLimit: Int,
    ) : NanoChatSession {
        override fun generateStream(request: GenerateContentRequest): Flow<NanoStreamChunk> =
            client.generateContentStream(request).map { resp ->
                val c = resp.candidates.firstOrNull()
                NanoStreamChunk(text = c?.text.orEmpty(), functionCalls = c?.functionCalls.orEmpty())
            }

        override suspend fun clearCaches() {
            runCatching { client.clearImplicitCaches() }
        }

        override fun close() {
            runCatching { client.close() }
        }
    }

    private companion object {
        const val TAG = "OndolSaem.AICore"
    }
}
