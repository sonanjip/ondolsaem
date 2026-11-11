package io.github.taehunkim.ondolsaem.engine

import com.google.mlkit.genai.common.FunctionArgValue
import com.google.mlkit.genai.common.FunctionCall
import com.google.mlkit.genai.common.GenAiException
import com.google.mlkit.genai.prompt.Content
import com.google.mlkit.genai.prompt.FunctionCallPart
import com.google.mlkit.genai.prompt.FunctionResponsePart
import com.google.mlkit.genai.prompt.GenerateContentRequest
import com.google.mlkit.genai.prompt.TextPart
import com.google.mlkit.genai.schema.tools.FunctionType
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.toList
import kotlinx.coroutines.runBlocking
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class GeminiNanoChatModelTest {

    private class FakeSession(
        override val supportsSystemPrompt: Boolean = true,
        override val supportsFunctionCalling: Boolean = true,
        override val tokenLimit: Int = 4096,
        private val responder: (request: GenerateContentRequest, callIndex: Int) -> List<NanoStreamChunk>,
    ) : NanoChatSession {
        val requests = mutableListOf<GenerateContentRequest>()
        var cleared = false
        var closed = false

        override fun generateStream(request: GenerateContentRequest): Flow<NanoStreamChunk> = flow {
            val idx = requests.size
            requests += request
            for (chunk in responder(request, idx)) {
                emit(chunk)
            }
        }

        override suspend fun clearCaches() {
            cleared = true
        }

        override fun close() {
            closed = true
        }
    }

    private fun fakeTool(name: String = "search", onExec: (JSONObject) -> String): ChatTool = object : ChatTool {
        override val name = name
        override val description = "Searches local files"
        override val parameters = """
            {
              "type": "object",
              "properties": {
                "query": { "type": "string", "description": "Search keywords" },
                "limit": { "type": "integer" },
                "tags": { "type": "array", "items": { "type": "string" } }
              },
              "required": ["query"]
            }
        """.trimIndent()

        override fun execute(args: JSONObject): String = onExec(args)
    }

    @Test fun convertsJsonSchemaToFunctionDeclaration() {
        val tool = fakeTool { "{}" }
        val decl = GeminiNanoChatModel.toFunctionDeclaration(tool)
        assertEquals("search", decl.name)
        assertEquals("Searches local files", decl.description)
        val params = decl.parameters!!
        assertEquals(FunctionType.OBJECT, params.type)
        assertEquals(listOf("query"), params.required)
        assertEquals(FunctionType.STRING, params.properties!!["query"]!!.type)
        assertEquals(FunctionType.INTEGER, params.properties!!["limit"]!!.type)
        assertEquals(FunctionType.ARRAY, params.properties!!["tags"]!!.type)
        assertEquals(FunctionType.STRING, params.properties!!["tags"]!!.items!!.type)
    }

    @Test fun convertsFunctionArgValuesToJson() {
        val json = GeminiNanoChatModel.functionArgsToJson(
            mapOf(
                "s" to FunctionArgValue.StringValue("hello"),
                "i" to FunctionArgValue.IntValue(3),
                "l" to FunctionArgValue.LongValue(99L),
                "d" to FunctionArgValue.DoubleValue(1.5),
                "b" to FunctionArgValue.BoolValue(true),
                "arr" to FunctionArgValue.ListValue(listOf(FunctionArgValue.StringValue("a"), FunctionArgValue.IntValue(2))),
                "nested" to FunctionArgValue.MapValue(mapOf("k" to FunctionArgValue.StringValue("v"))),
            ),
        )
        assertEquals("hello", json.getString("s"))
        assertEquals(3, json.getInt("i"))
        assertEquals(99L, json.getLong("l"))
        assertEquals(1.5, json.getDouble("d"), 1e-6)
        assertTrue(json.getBoolean("b"))
        assertEquals("a", json.getJSONArray("arr").getString(0))
        assertEquals(2, json.getJSONArray("arr").getInt(1))
        assertEquals("v", json.getJSONObject("nested").getString("k"))
    }

    @Test fun streamsMultiTurnWithImagesAndNativeSystemInstruction() = runBlocking {
        val session = FakeSession(supportsSystemPrompt = true) { _, _ ->
            listOf(NanoStreamChunk("Hello "), NanoStreamChunk("world!"))
        }
        val model = GeminiNanoChatModel(
            session = session,
            maxTokens = 8192,
            imagePartFactory = { bytes -> TextPart("[img:${bytes.size}]") },
        )

        val events = model.reply(
            conversationKey = "pkg|session1",
            system = "Be concise.",
            history = listOf(ChatTurn(TurnRole.USER, "Hi"), ChatTurn(TurnRole.ASSISTANT, "Hey!")),
            user = ChatTurn(TurnRole.USER, "Describe this", images = listOf(byteArrayOf(1, 2, 3))),
            tools = emptyList(),
        ).toList()

        assertEquals(listOf(ChatEvent.Token("Hello "), ChatEvent.Token("world!")), events)
        val req = session.requests.single()
        assertEquals("Be concise.", req.systemInstruction?.textString)
        assertNull(req.sessionId)
        assertFalse(req.enableThinking)
        assertEquals(3, req.contents.size)
        assertEquals(Content.Role.USER, req.contents[0].role)
        assertEquals(Content.Role.MODEL, req.contents[1].role)
        assertEquals(Content.Role.USER, req.contents[2].role)
        assertEquals(listOf("Describe this", "[img:3]"), req.contents[2].parts.map { (it as TextPart).textString })
        model.close()
        assertTrue(session.closed)
    }

    @Test fun injectsSystemPromptIntoFirstUserTurnWhenUnsupported() = runBlocking {
        val session = FakeSession(supportsSystemPrompt = false) { _, _ ->
            listOf(NanoStreamChunk("OK"))
        }
        val model = GeminiNanoChatModel(session, 4096) { null }
        model.reply("k", "Follow rules", emptyList(), ChatTurn(TurnRole.USER, "Question"), emptyList()).toList()

        val req = session.requests.single()
        assertNull(req.systemInstruction)
        val firstText = (req.contents.first().parts.first() as TextPart).textString
        assertTrue(firstText.startsWith("[System instructions]\nFollow rules"))
        assertTrue(firstText.endsWith("Question"))
    }

    @Test fun executesNativeFunctionCallLoopAndStreamsAnswer() = runBlocking {
        val session = FakeSession(supportsFunctionCalling = true) { _, idx ->
            when (idx) {
                0 -> listOf(
                    NanoStreamChunk(
                        text = "",
                        functionCalls = listOf(
                            FunctionCall(
                                name = "search",
                                arguments = mapOf("query" to FunctionArgValue.StringValue("receipt")),
                                id = "call_1",
                            ),
                        ),
                    ),
                )
                else -> listOf(NanoStreamChunk("Found "), NanoStreamChunk("1 receipt."))
            }
        }
        val model = GeminiNanoChatModel(session, 4096) { null }
        val tool = fakeTool { args -> "hits for ${args.getString("query")}" }

        val events = model.reply("k", "sys", emptyList(), ChatTurn(TurnRole.USER, "find receipt"), listOf(tool)).toList()

        assertEquals(2, session.requests.size)
        assertEquals(1, session.requests[0].dynamicTools.orEmpty().size)
        val round2Contents = session.requests[1].contents
        assertTrue(round2Contents[1].parts.any { it is FunctionCallPart })
        val respPart = round2Contents[2].parts.single() as FunctionResponsePart
        assertEquals("call_1", respPart.id)
        assertEquals("search", respPart.name)
        assertEquals("hits for receipt", respPart.response["result"])

        val ran = events.filterIsInstance<ChatEvent.ToolRan>().single()
        assertEquals("search", ran.name)
        assertEquals("hits for receipt", ran.result)
        assertEquals("Found 1 receipt.", events.filterIsInstance<ChatEvent.Token>().joinToString("") { it.text })
    }

    @Test fun executesTextToolFallbackAndHidesRawTagsFromStream() = runBlocking {
        val session = FakeSession(supportsFunctionCalling = false) { _, idx ->
            when (idx) {
                0 -> listOf(
                    NanoStreamChunk("Checking... "),
                    NanoStreamChunk("<tool_call>{\"name\":\"search\","),
                    NanoStreamChunk("\"arguments\":{\"query\":\"tax\"}}</tool_call>"),
                )
                else -> listOf(NanoStreamChunk("Your 2025 tax PDF is ready."))
            }
        }
        val model = GeminiNanoChatModel(session, 4096) { null }
        val tool = fakeTool { args -> "tax_2025.pdf (${args.getString("query")})" }

        val events = model.reply("k", "sys", emptyList(), ChatTurn(TurnRole.USER, "tax?"), listOf(tool)).toList()

        assertTrue(session.requests[0].dynamicTools.isNullOrEmpty())
        val text = events.filterIsInstance<ChatEvent.Token>().joinToString("") { it.text }
        assertEquals("Checking... Your 2025 tax PDF is ready.", text)
        assertFalse(text.contains("<tool_call>"))
        val ran = events.filterIsInstance<ChatEvent.ToolRan>().single()
        assertEquals("tax_2025.pdf (tax)", ran.result)
    }

    @Test fun fallsBackToTextToolsWhenNativeToolsFailWithNotSupported() = runBlocking {
        val session = FakeSession(supportsFunctionCalling = true) { req, _ ->
            if (!req.dynamicTools.isNullOrEmpty()) {
                throw GenAiException("not supported", null, GenAiException.ErrorCode.NOT_SUPPORTED)
            }
            listOf(NanoStreamChunk("Fallback answer"))
        }
        val model = GeminiNanoChatModel(session, 4096) { null }
        val tool = fakeTool { "ok" }

        val events = model.reply("k", "sys", emptyList(), ChatTurn(TurnRole.USER, "hi"), listOf(tool)).toList()
        assertEquals(2, session.requests.size)
        assertEquals("Fallback answer", events.filterIsInstance<ChatEvent.Token>().joinToString("") { it.text })
    }

    @Test fun streamTextBufferReleasesNormalLessThanCharactersImmediately() {
        val buf = GeminiNanoChatModel.StreamTextBuffer()
        val out = StringBuilder()
        for (ch in listOf("3 ", "<", " 5 and <b>ok</b>")) {
            out.append(buf.append(ch))
        }
        out.append(buf.flushAll())
        assertEquals("3 < 5 and <b>ok</b>", out.toString())
    }
}
