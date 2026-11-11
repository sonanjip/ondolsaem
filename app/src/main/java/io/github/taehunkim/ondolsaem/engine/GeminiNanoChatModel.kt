package io.github.taehunkim.ondolsaem.engine

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.Log
import com.google.mlkit.genai.common.FunctionArgValue
import com.google.mlkit.genai.common.FunctionCall
import com.google.mlkit.genai.common.GenAiException
import com.google.mlkit.genai.prompt.Content
import com.google.mlkit.genai.prompt.FunctionCallPart
import com.google.mlkit.genai.prompt.FunctionResponsePart
import com.google.mlkit.genai.prompt.GenerateContentRequest
import com.google.mlkit.genai.prompt.ImagePart
import com.google.mlkit.genai.prompt.Part
import com.google.mlkit.genai.prompt.SystemInstruction
import com.google.mlkit.genai.prompt.TextPart
import com.google.mlkit.genai.schema.tools.FunctionDeclaration
import com.google.mlkit.genai.schema.tools.FunctionSchema
import com.google.mlkit.genai.schema.tools.FunctionType
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.channelFlow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.launch
import org.json.JSONArray
import org.json.JSONObject
import java.util.concurrent.atomic.AtomicBoolean

/**
 * [ChatModel] backed by Gemini Nano via Android AICore (`com.google.mlkit:genai-prompt`).
 *
 * Weights and KV cache reside inside the system AICore process rather than OndolSaem's heap, keeping host process
 * memory overhead minimal while supporting multimodal images and multi-round tool execution.
 */
class GeminiNanoChatModel internal constructor(
    private val session: NanoChatSession,
    private val maxTokens: Int,
    private val imagePartFactory: (ByteArray) -> Part? = ::defaultImagePart,
) : ChatModel {

    override val supportsImages: Boolean = true
    override val backend: String = BACKEND_NAME

    private val closed = AtomicBoolean(false)
    private val cleanupScope = CoroutineScope(SupervisorJob() + Dispatchers.IO)

    override fun reply(
        conversationKey: String,
        system: String,
        history: List<ChatTurn>,
        user: ChatTurn,
        tools: List<ChatTool>,
    ): Flow<ChatEvent> = channelFlow {
        check(!closed.get()) { "GeminiNanoChatModel is closed" }

        var useNativeTools = session.supportsFunctionCalling && tools.isNotEmpty()
        var emittedAnything = false

        while (true) {
            try {
                runConversationLoop(
                    system = system,
                    history = history,
                    user = user,
                    tools = tools,
                    useNativeTools = useNativeTools,
                    onEmit = { event ->
                        emittedAnything = true
                        send(event)
                    },
                )
                break
            } catch (e: GenAiException) {
                // If AICore rejects native function declarations at runtime, fall back once to prompt-based tool calls.
                if (useNativeTools && !emittedAnything && isUnsupportedToolError(e)) {
                    Log.w(TAG, "Native function calling rejected by AICore (code=${e.errorCode}); falling back to prompt tools")
                    useNativeTools = false
                    continue
                }
                throw e
            }
        }
    }.flowOn(Dispatchers.Default)

    private suspend fun runConversationLoop(
        system: String,
        history: List<ChatTurn>,
        user: ChatTurn,
        tools: List<ChatTool>,
        useNativeTools: Boolean,
        onEmit: suspend (ChatEvent) -> Unit,
    ) {
        val useTextTools = !useNativeTools && tools.isNotEmpty()
        val effectiveSystem = buildEffectiveSystem(system, tools, useTextTools)
        val contents = buildInitialContents(effectiveSystem, history, user)
        val declarations = if (useNativeTools) tools.map(::toFunctionDeclaration) else emptyList()

        for (round in 0 until MAX_TOOL_ROUNDS) {
            val request = buildRequest(effectiveSystem, contents, declarations)
            val roundText = StringBuilder()
            val streamBuffer = StreamTextBuffer()
            val nativeCalls = LinkedHashSet<FunctionCall>()

            session.generateStream(request).collect { chunk ->
                val delta = chunk.text
                if (delta.isNotEmpty()) {
                    roundText.append(delta)
                    val safeChunk = streamBuffer.append(delta)
                    if (safeChunk.isNotEmpty()) onEmit(ChatEvent.Token(safeChunk))
                }
                for (fc in chunk.functionCalls) {
                    nativeCalls.add(fc)
                }
            }

            if (nativeCalls.isNotEmpty()) {
                val flushed = streamBuffer.flushAll()
                if (flushed.isNotEmpty()) onEmit(ChatEvent.Token(flushed))

                val modelParts = buildList<Part> {
                    if (roundText.isNotEmpty()) add(TextPart(roundText.toString()))
                    nativeCalls.forEach { add(FunctionCallPart(it)) }
                }
                contents.add(buildContent(Content.Role.MODEL, modelParts))

                val responseParts = nativeCalls.map { call ->
                    val args = functionArgsToJson(call.arguments)
                    val result = executeTool(tools, call.name, args)
                    onEmit(ChatEvent.ToolRan(call.name, args.toString(), result))
                    FunctionResponsePart(call.id, call.name, mapOf("result" to result))
                }
                contents.add(buildContent(Content.Role.USER, responseParts))
                continue
            }

            val fullRoundText = roundText.toString()
            val textCalls = if (tools.isNotEmpty()) parseTextToolCalls(fullRoundText) else emptyList()
            if (textCalls.isNotEmpty()) {
                // Emit only any trailing safe text before the first tool tag that was still held in the buffer.
                val beforeTag = streamBuffer.flushPrefixBeforeToolTag()
                if (beforeTag.isNotEmpty()) onEmit(ChatEvent.Token(beforeTag))

                contents.add(buildContent(Content.Role.MODEL, listOf(TextPart(fullRoundText))))
                val resultsText = StringBuilder()
                for (tc in textCalls) {
                    val result = executeTool(tools, tc.name, tc.args)
                    onEmit(ChatEvent.ToolRan(tc.name, tc.args.toString(), result))
                    resultsText.append("[Tool result for ").append(tc.name).append("]:\n").append(result).append("\n\n")
                }
                contents.add(
                    buildContent(
                        Content.Role.USER,
                        listOf(TextPart("${resultsText}Use the tool results above to answer the user's request.")),
                    ),
                )
                continue
            }

            val tail = streamBuffer.flushAll()
            if (tail.isNotEmpty()) onEmit(ChatEvent.Token(tail))
            return
        }
    }

    private fun buildEffectiveSystem(system: String, tools: List<ChatTool>, useTextTools: Boolean): String {
        if (!useTextTools || tools.isEmpty()) return system
        val toolGuide = buildString {
            append("You have access to the following tools. When you need to call a tool, respond with ONLY ")
            append("<tool_call>{\"name\":\"tool_name\",\"arguments\":{...}}</tool_call> and wait for the result.\n")
            append("Tools:\n")
            for (t in tools) {
                append("- ").append(t.name).append(": ").append(t.description)
                    .append(" Parameters JSON schema: ").append(t.parameters).append('\n')
            }
        }
        return if (system.isBlank()) toolGuide else "$system\n\n$toolGuide"
    }

    private fun buildInitialContents(effectiveSystem: String, history: List<ChatTurn>, user: ChatTurn): MutableList<Content> {
        val list = ArrayList<Content>(history.size + 1)
        val prefixSystem = if (!session.supportsSystemPrompt && effectiveSystem.isNotBlank()) {
            "[System instructions]\n$effectiveSystem\n\n"
        } else {
            ""
        }
        var appliedPrefix = false
        for (turn in history) {
            val role = if (turn.role == TurnRole.USER) Content.Role.USER else Content.Role.MODEL
            val text = if (!appliedPrefix && role == Content.Role.USER && prefixSystem.isNotEmpty()) {
                appliedPrefix = true
                prefixSystem + turn.text
            } else {
                turn.text
            }
            list.add(buildContent(role, listOf(TextPart(text))))
        }

        val userText = if (!appliedPrefix && prefixSystem.isNotEmpty()) prefixSystem + user.text else user.text
        val userParts = buildList<Part> {
            add(TextPart(userText))
            for (img in user.images) {
                imagePartFactory(img)?.let(::add)
            }
        }
        list.add(buildContent(Content.Role.USER, userParts))
        return list
    }

    private fun buildRequest(
        effectiveSystem: String,
        contents: List<Content>,
        declarations: List<FunctionDeclaration>,
    ): GenerateContentRequest {
        val builder = GenerateContentRequest.Builder(contents)
        if (session.supportsSystemPrompt && effectiveSystem.isNotBlank()) {
            builder.systemInstruction = SystemInstruction(effectiveSystem)
        }
        builder.enableThinking = false
        val budget = minOf(maxTokens, session.tokenLimit)
        builder.maxOutputTokens = minOf(DEFAULT_MAX_OUTPUT_TOKENS, (budget / 2).coerceAtLeast(256))
        if (declarations.isNotEmpty()) {
            builder.setTools(declarations)
        }
        return builder.build()
    }

    private fun executeTool(tools: List<ChatTool>, name: String, args: JSONObject): String {
        val tool = tools.firstOrNull { it.name == name } ?: return "Error: unknown tool $name"
        return try {
            tool.execute(args)
        } catch (e: CancellationException) {
            throw e
        } catch (e: Exception) {
            "Error: ${e.message}"
        }
    }

    private fun isUnsupportedToolError(e: GenAiException): Boolean =
        e.errorCode == GenAiException.ErrorCode.NOT_SUPPORTED ||
            e.errorCode == GenAiException.ErrorCode.REQUEST_PROCESSING_ERROR

    override fun reset() {
        if (closed.get()) return
        cleanupScope.launch { session.clearCaches() }
    }

    override fun close() {
        if (!closed.compareAndSet(false, true)) return
        runCatching { session.close() }
    }

    /**
     * Holds back only potential `<tool` / `<tool_call` markup prefixes while streaming plain prose immediately.
     */
    internal class StreamTextBuffer {
        private val pending = StringBuilder()
        private var insideToolTag = false

        fun append(chunk: String): String {
            pending.append(chunk)
            if (insideToolTag) {
                val str = pending.toString()
                val endIdx = findClosingTagEnd(str)
                if (endIdx >= 0) {
                    // Drop the completed tool tag from user-visible text and continue streaming after it.
                    pending.delete(0, endIdx)
                    insideToolTag = false
                    return flushSafePrefix()
                }
                return ""
            }
            return flushSafePrefix()
        }

        fun flushPrefixBeforeToolTag(): String {
            val str = pending.toString()
            val idx = str.indexOf('<')
            if (idx <= 0) {
                if (idx == 0) pending.clear()
                return ""
            }
            val prefix = str.substring(0, idx)
            pending.clear()
            insideToolTag = false
            return prefix
        }

        fun flushAll(): String {
            if (insideToolTag) {
                // Unclosed `<tool` tag at end of stream: keep text only if no tool call marker was present.
                val s = pending.toString()
                pending.clear()
                insideToolTag = false
                return if (s.startsWith("<tool")) "" else s
            }
            val out = pending.toString()
            pending.clear()
            return out
        }

        private fun flushSafePrefix(): String {
            val out = StringBuilder()
            while (pending.isNotEmpty()) {
                val lt = pending.indexOf('<')
                if (lt < 0) {
                    out.append(pending)
                    pending.clear()
                    break
                }
                if (lt > 0) {
                    out.append(pending, 0, lt)
                    pending.delete(0, lt)
                }
                val rem = pending.toString()
                if (isToolTagPrefix(rem)) {
                    if (rem.startsWith("<tool_call") || rem.startsWith("<tool ") || rem.startsWith("<tool>")) {
                        insideToolTag = true
                        val endIdx = findClosingTagEnd(rem)
                        if (endIdx >= 0) {
                            pending.delete(0, endIdx)
                            insideToolTag = false
                            continue
                        }
                    }
                    // Wait for more characters to decide whether `<` starts a tool call tag.
                    break
                } else {
                    // Confirmed not a tool tag (e.g. `<10` or `<a>`); release `<` and continue.
                    out.append('<')
                    pending.deleteCharAt(0)
                }
            }
            return out.toString()
        }

        private fun isToolTagPrefix(s: String): Boolean =
            TOOL_PREFIXES.any { prefix ->
                if (s.length <= prefix.length) prefix.startsWith(s) else s.startsWith(prefix)
            }

        private fun findClosingTagEnd(s: String): Int {
            for (closing in CLOSING_TAGS) {
                val idx = s.indexOf(closing)
                if (idx >= 0) return idx + closing.length
            }
            return -1
        }

        private companion object {
            val TOOL_PREFIXES = listOf("<tool_call>", "<tool_call ", "<tool ", "<tool>")
            val CLOSING_TAGS = listOf("</tool_call>", "</tool>")
        }
    }

    internal data class ParsedTextToolCall(val name: String, val args: JSONObject)

    companion object {
        private const val TAG = "OndolSaem.GeminiNano"
        const val BACKEND_NAME = "AICore (NPU)"
        const val MAX_CONTEXT_TOKENS = 4096
        private const val DEFAULT_MAX_OUTPUT_TOKENS = 1024
        private const val MAX_TOOL_ROUNDS = 8
        private const val MAX_IMAGE_SIDE = 1024

        suspend fun load(gateway: AiCoreGateway, maxTokens: Int): GeminiNanoChatModel {
            val session = gateway.openChatSession()
            return GeminiNanoChatModel(session, minOf(maxTokens, session.tokenLimit))
        }

        internal fun buildContent(role: Int, parts: List<Part>): Content =
            Content.Builder().apply {
                role(role)
                parts.forEach(::addPart)
            }.build()

        internal fun defaultImagePart(bytes: ByteArray): Part? {
            if (bytes.isEmpty()) return null
            val opts = BitmapFactory.Options().apply { inJustDecodeBounds = true }
            BitmapFactory.decodeByteArray(bytes, 0, bytes.size, opts)
            if (opts.outWidth > 0 && opts.outHeight > 0) {
                val maxDim = maxOf(opts.outWidth, opts.outHeight)
                val sample = if (maxDim > MAX_IMAGE_SIDE) Integer.highestOneBit(maxDim / MAX_IMAGE_SIDE).coerceAtLeast(1) else 1
                val decOpts = BitmapFactory.Options().apply { inSampleSize = sample }
                val bmp = BitmapFactory.decodeByteArray(bytes, 0, bytes.size, decOpts)
                if (bmp != null) {
                    val scaled = scaleDownIfNeeded(bmp, MAX_IMAGE_SIDE)
                    if (scaled !== bmp) bmp.recycle()
                    return ImagePart(scaled)
                }
            }
            return runCatching { ImagePart(bytes) }.getOrNull()
        }

        private fun scaleDownIfNeeded(bmp: Bitmap, maxSide: Int): Bitmap {
            val w = bmp.width
            val h = bmp.height
            if (w <= maxSide && h <= maxSide) return bmp
            val ratio = maxSide.toFloat() / maxOf(w, h)
            return Bitmap.createScaledBitmap(bmp, (w * ratio).toInt().coerceAtLeast(1), (h * ratio).toInt().coerceAtLeast(1), true)
        }

        internal fun toFunctionDeclaration(tool: ChatTool): FunctionDeclaration {
            val schemaJson = runCatching { JSONObject(tool.parameters) }.getOrElse { JSONObject() }
            return FunctionDeclaration(
                name = tool.name,
                description = tool.description,
                parameters = toFunctionSchema(schemaJson),
            )
        }

        internal fun toFunctionSchema(json: JSONObject): FunctionSchema {
            val typeStr = json.optString("type").lowercase()
            val type = when (typeStr) {
                "string" -> FunctionType.STRING
                "number" -> FunctionType.NUMBER
                "integer" -> FunctionType.INTEGER
                "boolean" -> FunctionType.BOOLEAN
                "array" -> FunctionType.ARRAY
                "object" -> FunctionType.OBJECT
                "null" -> FunctionType.NULL
                else -> if (json.has("properties")) FunctionType.OBJECT else FunctionType.UNSPECIFIED
            }
            val propsObj = json.optJSONObject("properties")
            val properties = propsObj?.let { p ->
                p.keys().asSequence().associateWith { k -> toFunctionSchema(p.getJSONObject(k)) }
            }
            val reqArr = json.optJSONArray("required")
            val required = reqArr?.let { arr -> (0 until arr.length()).map { arr.getString(it) } }
            val enumArr = json.optJSONArray("enum")
            val enumVals = enumArr?.let { arr -> (0 until arr.length()).map { arr.optString(it) } }
            val items = json.optJSONObject("items")?.let(::toFunctionSchema)
            val anyOfArr = json.optJSONArray("anyOf")
            val anyOf = anyOfArr?.let { arr ->
                (0 until arr.length()).mapNotNull { arr.optJSONObject(it)?.let(::toFunctionSchema) }
            }

            return FunctionSchema(
                type = type,
                description = json.optString("description").ifEmpty { null },
                properties = properties,
                required = required,
                enum = enumVals,
                format = json.optString("format").ifEmpty { null },
                nullable = if (json.has("nullable")) json.optBoolean("nullable") else null,
                items = items,
                minItems = if (json.has("minItems")) json.optLong("minItems") else null,
                maxItems = if (json.has("maxItems")) json.optLong("maxItems") else null,
                minimum = if (json.has("minimum")) json.optDouble("minimum") else null,
                maximum = if (json.has("maximum")) json.optDouble("maximum") else null,
                minLength = if (json.has("minLength")) json.optLong("minLength") else null,
                maxLength = if (json.has("maxLength")) json.optLong("maxLength") else null,
                pattern = json.optString("pattern").ifEmpty { null },
                anyOf = anyOf,
            )
        }

        internal fun functionArgsToJson(args: Map<String, FunctionArgValue>): JSONObject {
            val out = JSONObject()
            for ((k, v) in args) {
                out.put(k, argValueToAny(v))
            }
            return out
        }

        private fun argValueToAny(value: FunctionArgValue): Any = when (value) {
            is FunctionArgValue.StringValue -> value.value
            is FunctionArgValue.IntValue -> value.value
            is FunctionArgValue.LongValue -> value.value
            is FunctionArgValue.DoubleValue -> value.value
            is FunctionArgValue.BoolValue -> value.value
            is FunctionArgValue.ListValue -> JSONArray(value.values.map(::argValueToAny))
            is FunctionArgValue.MapValue -> functionArgsToJson(value.map)
        }

        internal fun parseTextToolCalls(text: String): List<ParsedTextToolCall> {
            val calls = ArrayList<ParsedTextToolCall>()
            val rJson = Regex("<(?:tool_call|tool)>(.*?)</(?:tool_call|tool)>", RegexOption.DOT_MATCHES_ALL)
            for (m in rJson.findAll(text)) {
                val body = m.groupValues[1].trim()
                val json = runCatching { JSONObject(body) }.getOrNull()
                if (json != null) {
                    val name = json.optString("name").ifEmpty { json.optString("function") }.trim()
                    val args = json.optJSONObject("arguments") ?: json.optJSONObject("parameters") ?: JSONObject()
                    if (name.isNotEmpty()) calls += ParsedTextToolCall(name, args)
                }
            }
            if (calls.isNotEmpty()) return calls

            val rNamed = Regex("<tool\\s+name=[\"']([^\"']+)[\"']\\s*>(.*?)</tool>", RegexOption.DOT_MATCHES_ALL)
            for (m in rNamed.findAll(text)) {
                val name = m.groupValues[1].trim()
                val body = m.groupValues[2].trim()
                val args = runCatching { JSONObject(body) }.getOrElse { JSONObject() }
                if (name.isNotEmpty()) calls += ParsedTextToolCall(name, args)
            }
            return calls
        }
    }
}
