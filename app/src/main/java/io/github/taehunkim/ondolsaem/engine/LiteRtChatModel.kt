package io.github.taehunkim.ondolsaem.engine

import android.os.Build
import android.util.Log
import com.google.ai.edge.litertlm.Backend
import com.google.ai.edge.litertlm.Content
import com.google.ai.edge.litertlm.Contents
import com.google.ai.edge.litertlm.Conversation
import com.google.ai.edge.litertlm.ConversationConfig
import com.google.ai.edge.litertlm.Engine
import com.google.ai.edge.litertlm.EngineConfig
import com.google.ai.edge.litertlm.ExperimentalApi
import com.google.ai.edge.litertlm.Message
import com.google.ai.edge.litertlm.OpenApiTool
import com.google.ai.edge.litertlm.tool
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.channelFlow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.io.File

@OptIn(ExperimentalApi::class)
/** [ChatModel] on LiteRT-LM (Gemma 4, Qwen, ...), with the model's tool calling executing [ChatTool]s. */
class LiteRtChatModel private constructor(
    private val engine: Engine,
    override val supportsImages: Boolean,
    private val maxTokens: Int,
    override val backend: String = "",
) : ChatModel {
    private val lock = Any()
    private var conversation: Conversation? = null
    private var conversationKey: String? = null
    private var knownTurns = 0

    @Volatile private var listener: ((ChatEvent) -> Unit)? = null

    override fun reply(conversationKey: String, system: String, history: List<ChatTurn>, user: ChatTurn, tools: List<ChatTool>): Flow<ChatEvent> =
        channelFlow {
            val conv = prepare(conversationKey, system, history, tools)
            listener = { trySend(it) }
            var finished = false
            try {
                val message = if (user.images.isNotEmpty() && supportsImages) {
                    Message.user(Contents.of(listOf<Content>(Content.Text(user.text)) + user.images.map { Content.ImageBytes(it) }))
                } else Message.user(user.text)
                conv.sendMessageAsync(message).collect { m ->
                    val text = m.contents.contents.filterIsInstance<Content.Text>().joinToString("") { it.text }
                    if (text.isNotEmpty()) send(ChatEvent.Token(text))
                }
                finished = true
            } finally {
                listener = null
                synchronized(lock) {
                    if (finished) {
                        knownTurns += 2
                        // Replying keeps growing the model's own context; start over from the (trimmed) history before it overflows.
                        if (conv.getTokenCount() > maxTokens * REBUILD_AT) knownTurns = -1
                    } else {
                        runCatching { conv.cancelProcess() }
                        knownTurns = -1 // partial answer: the stored history is the truth
                    }
                }
            }
        }.flowOn(Dispatchers.Default)

    private fun prepare(key: String, system: String, history: List<ChatTurn>, tools: List<ChatTool>): Conversation = synchronized(lock) {
        val reuse = conversation?.takeIf { it.isAlive && key == conversationKey && knownTurns == history.size }
        if (reuse != null) return reuse
        conversation?.let { runCatching { it.close() } }
        val initial = history.map { t ->
            if (t.role == TurnRole.USER) Message.user(t.text) else Message.model(t.text)
        }
        val config = ConversationConfig(
            systemInstruction = Contents.of(system),
            initialMessages = initial,
            tools = tools.map { tool(wrap(it)) },
            automaticToolCalling = true,
        )
        engine.createConversation(config).also {
            conversation = it; conversationKey = key; knownTurns = history.size
        }
    }

    private fun wrap(t: ChatTool) = object : OpenApiTool {
        override fun getToolDescriptionJsonString(): String =
            JSONObject().put("name", t.name).put("description", t.description).put("parameters", JSONObject(t.parameters)).toString()

        override fun execute(paramsJsonString: String): String {
            val args = runCatching { JSONObject(paramsJsonString) }.getOrElse { JSONObject() }
            val result = try { t.execute(args) } catch (e: Exception) { "Error: ${e.message}" }
            Log.d(TAG, "tool ${t.name}($paramsJsonString) -> ${result.take(200)}")
            listener?.invoke(ChatEvent.ToolRan(t.name, paramsJsonString, result))
            return result
        }
    }

    override fun reset() = synchronized(lock) {
        conversation?.let { runCatching { it.close() } }
        conversation = null; conversationKey = null; knownTurns = 0
    }

    override fun close() {
        reset()
        runCatching { engine.close() }
    }

    companion object {
        private const val TAG = "OndolSaem.Chat"
        private const val REBUILD_AT = 0.7

        private val isEmulator: Boolean
            get() = Build.FINGERPRINT.contains("generic") || Build.HARDWARE.contains("ranchu") || Build.HARDWARE.contains("goldfish")

        /**
         * Loads the model (can take a minute; off the main thread). With [npu], its NPU build first (images on the GPU,
         * as AI Edge Gallery does); then the GPU on real devices, the CPU otherwise. An NPU build has its context
         * length fixed when it was compiled: a larger [maxTokens] fails there and falls back to the GPU.
         */
        suspend fun load(
            modelFile: File, cacheDir: File, vision: Boolean, maxTokens: Int, npu: NpuTarget? = null,
            /** True: the NPU build only, failing instead of falling back (NPU self-check). */
            npuOnly: Boolean = false,
            audio: Boolean = false,
        ): LiteRtChatModel =
            withContext(Dispatchers.Default) {
                cacheDir.mkdirs()
                class Attempt(val file: File, val main: () -> Backend, val vision: () -> Backend)
                val plain: List<() -> Backend> =
                    if (isEmulator) listOf({ Backend.CPU() }) else listOf({ Backend.GPU() }, { Backend.CPU() })
                val attempts = listOfNotNull(
                    npu?.let { n -> Attempt(n.file, { Backend.NPU(nativeLibraryDir = n.libDir) }, { Backend.GPU() }) },
                ) + if (npuOnly) emptyList() else plain.map { b -> Attempt(modelFile, b, b) }
                var last: Throwable? = null
                for (a in attempts) {
                    val backend = a.main()
                    val engine = Engine(
                        EngineConfig(
                            modelPath = a.file.absolutePath,
                            backend = backend,
                            visionBackend = if (vision) a.vision() else null,
                            audioBackend = if (audio) Backend.CPU() else null,
                            maxNumTokens = maxTokens,
                            maxNumImages = if (vision) 2 else null,
                            cacheDir = cacheDir.absolutePath,
                        ),
                    )
                    try {
                        val t0 = System.currentTimeMillis()
                        engine.initialize()
                        if (backend is Backend.NPU && NpuLog.registered(t0 - 50) == false) {
                            throw IllegalStateException("the NPU accelerator was not registered: LiteRT would run the NPU build on the CPU")
                        }
                        Log.i(TAG, "loaded ${a.file.name} on ${backend.name}")
                        return@withContext LiteRtChatModel(engine, vision, maxTokens, backend.name)
                    } catch (t: Throwable) {
                        Log.w(TAG, "init of ${a.file.name} failed on ${backend.name}", t)
                        runCatching { engine.close() }
                        last = t
                    }
                }
                throw IllegalStateException("Could not load the chat model", last)
            }
    }
}
