package io.github.taehunkim.ondolsaem

import android.graphics.Bitmap
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import io.github.taehunkim.ondolsaem.engine.ChatEvent
import io.github.taehunkim.ondolsaem.engine.ChatModel
import io.github.taehunkim.ondolsaem.engine.ChatTool
import io.github.taehunkim.ondolsaem.engine.ChatTurn
import io.github.taehunkim.ondolsaem.engine.Cue
import io.github.taehunkim.ondolsaem.engine.DetectedFace
import io.github.taehunkim.ondolsaem.engine.Embedder
import io.github.taehunkim.ondolsaem.catalog.ModelCategories
import io.github.taehunkim.ondolsaem.engine.Decider
import io.github.taehunkim.ondolsaem.engine.FaceEngine
import io.github.taehunkim.ondolsaem.engine.OcrEngine
import io.github.taehunkim.ondolsaem.engine.Span
import io.github.taehunkim.ondolsaem.engine.SpeechTools
import io.github.taehunkim.ondolsaem.runtime.Residency
import io.github.taehunkim.ondolsaem.sdk.ChatStreamEvent
import io.github.taehunkim.ondolsaem.sdk.HistoryTurn
import io.github.taehunkim.ondolsaem.sdk.HostCallException
import io.github.taehunkim.ondolsaem.sdk.HostUnavailableException
import io.github.taehunkim.ondolsaem.sdk.OndolSaem
import io.github.taehunkim.ondolsaem.sdk.OndolSaem.Categories
import io.github.taehunkim.ondolsaem.sdk.OndolSaemClient
import io.github.taehunkim.ondolsaem.sdk.ToolSpec
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitCancellation
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.channelFlow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.take
import kotlinx.coroutines.flow.toList
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeout
import org.json.JSONObject
import org.junit.After
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import java.io.ByteArrayOutputStream
import java.io.File

/**
 * Talks to the real bound service through the SDK client, with fake engines in place of the models (the test runs
 * in the host's own process, which the caller check always allows). Does not depend on downloaded models.
 */
@RunWith(AndroidJUnit4::class)
class HostServiceTest {
    private val ctx = InstrumentationRegistry.getInstrumentation().targetContext
    private val container get() = (ctx.applicationContext as OndolSaemApp).container
    private val runtime get() = container.runtime
    private lateinit var client: OndolSaemClient

    private class FakeEmbedder(override val modelId: String) : Embedder {
        var closed = false
        override suspend fun embedImage(jpeg: ByteArray) = floatArrayOf(jpeg.size.toFloat(), jpeg.last().toFloat())
        override suspend fun embedDocument(text: String, title: String?) = floatArrayOf(2f, text.length.toFloat())
        override suspend fun embedQuery(text: String) = floatArrayOf(1f, text.length.toFloat())
        override suspend fun embedImages(jpegs: List<ByteArray>) = floatArrayOf(jpegs.size.toFloat())
        override fun close() { closed = true }
    }

    /** Calls the first tool, then streams the tool's result back word by word (or slowly forever when asked). */
    private class FakeChat : ChatModel {
        override val supportsImages = true
        var cancelled = CompletableDeferred<Unit>()
        var lastKey: String? = null
        var lastHistory: List<ChatTurn> = emptyList()
        override fun reply(conversationKey: String, system: String, history: List<ChatTurn>, user: ChatTurn, tools: List<ChatTool>): Flow<ChatEvent> =
            channelFlow {
                lastKey = conversationKey; lastHistory = history
                if (user.text == "forever") {
                    try { send(ChatEvent.Token("…")); awaitCancellation() } finally { cancelled.complete(Unit) }
                }
                val tool = tools.firstOrNull()
                if (tool != null) {
                    val args = JSONObject().put("q", user.text).toString()
                    val result = tool.execute(JSONObject(args))
                    send(ChatEvent.ToolRan(tool.name, args, result))
                    result.split(" ").forEach { send(ChatEvent.Token("$it ")) }
                } else {
                    send(ChatEvent.Token("images=${user.images.size} "))
                    send(ChatEvent.Token("system=$system"))
                }
            }
        override fun reset() {}
        override fun close() {}
    }

    private class FakeSpeech(override val canTranscribe: Boolean) : SpeechTools {
        override suspend fun speechSpans(pcm: File) = listOf(Span(0, pcm.length() / 32), Span(5_000, 6_000))
        override suspend fun transcribeWithLanguage(pcm: File, fromMs: Long, language: String, onCue: suspend (Cue) -> Unit) {
            if (language == "auto") transcribe(pcm, fromMs, onCue)
            else onCue(Cue(fromMs, fromMs + 900, "language:$language"))
        }
        override suspend fun transcribe(pcm: File, fromMs: Long, onCue: suspend (Cue) -> Unit) {
            for (i in 0 until 5) onCue(Cue(fromMs + i * 1000, fromMs + i * 1000 + 900, "cue$i"))
        }
    }

    private lateinit var embedder: FakeEmbedder
    private val chat = FakeChat()
    private var embedLoads = 0

    @Before fun setUp() = runBlocking {
        runtime.readyOverride = { true }
        runtime.embedderFactory = { spec, _, _ -> embedLoads++; FakeEmbedder(spec.id).also { embedder = it } }
        runtime.chatFactory = { _, _, _, _ -> chat }
        runtime.ocrFactory = { spec -> object : OcrEngine {
            override val modelId = spec.id
            override suspend fun recognize(bitmap: Bitmap) = "${bitmap.width}x${bitmap.height}"
        } }
        runtime.speechFactory = { dir -> FakeSpeech(dir != null) }
        runtime.faceFactory = { spec, _ -> object : FaceEngine {
            override val modelId = spec.id
            // one face per 20 px of width, vector = its index
            override suspend fun detect(bitmap: Bitmap) = (0 until bitmap.width / 20).map { i ->
                DetectedFace(i * 0.1f, 0.1f, i * 0.1f + 0.1f, 0.5f, 0.9f, floatArrayOf(i.toFloat(), 1f))
            }
        } }
        runtime.deciderFactory = { spec, _, _ -> object : Decider {
            override val modelId = spec.id
            // yes when the premise contains the hypothesis' last word
            override fun entail(premise: String, hypotheses: List<String>) =
                hypotheses.map { if (premise.contains(it.trimEnd('.').substringAfterLast(' '))) 0.9f else 0.1f }.toFloatArray()
            override fun entailImage(imageBytes: ByteArray, premise: String, hypotheses: List<String>) =
                hypotheses.map { if (imageBytes.isNotEmpty()) 0.95f else 0.05f }.toFloatArray()
        } }
        runtime.imageGenFactory = { spec, _ -> object : io.github.taehunkim.ondolsaem.engine.ImageGenEngine {
            override val modelId = spec.id
            override suspend fun generateImage(prompt: String, seed: Long): ByteArray =
                byteArrayOf(0x89.toByte(), 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A)
        } }
        for (c in listOf(Categories.EMBEDDING, Categories.LLM, Categories.OCR, Categories.ASR, Categories.FACE, Categories.DECISION, Categories.IMAGE_GEN)) {
            container.settings.setResidency(c, Residency.IDLE_2M)
            runtime.unload(c)
        }
        container.settings.setActiveModel(Categories.EMBEDDING, "eg2-740m")
        client = OndolSaemClient(ctx, idleUnbindMs = 500)
    }

    @After fun tearDown() = runBlocking {
        container.apiVersion = OndolSaem.API_VERSION
        container.minClientApi = HostContainer.MIN_CLIENT_API
        runtime.readyOverride = null
        client.disconnect()
        for (c in listOf(Categories.EMBEDDING, Categories.LLM, Categories.OCR, Categories.ASR, Categories.FACE, Categories.DECISION, Categories.IMAGE_GEN)) runtime.unload(c)
    }

    @Test fun reportsApiAndCategories() = runBlocking {
        assertEquals(OndolSaem.API_VERSION, client.apiVersion())
        val cats = client.categories().map { it.category }
        assertEquals(listOf(Categories.EMBEDDING, Categories.LLM, Categories.OCR, Categories.ASR, Categories.FACE, Categories.DECISION, Categories.IMAGE_GEN), cats)
        val emb = client.activeModel(Categories.EMBEDDING)!!
        assertEquals("eg2-740m", emb.id)
        assertTrue(emb.ready)
        assertNull(client.activeModel("no-such-category"))
    }

    @Test fun embedsTextWithModelIdentity() = runBlocking {
        val q = client.embedQuery("hello")
        assertArrayEquals(floatArrayOf(1f, 5f), q.vector, 0f)
        assertEquals("eg2-740m", q.modelId)
        assertEquals(1, q.modelVersion)
        val d = client.embedDocument("abc", "t")
        assertArrayEquals(floatArrayOf(2f, 3f), d.vector, 0f)
        assertEquals("loaded once for both calls", 1, embedLoads)
    }

    @Test fun passesLargeInputsAsFileDescriptors() = runBlocking {
        val big = ByteArray(3_000_000) { (it % 7).toByte() } // well over Binder's 1 MB limit
        val r = client.embedImage(big)
        assertArrayEquals(floatArrayOf(3_000_000f, big.last().toFloat()), r.vector, 0f)
        assertArrayEquals(floatArrayOf(3f), client.embedImages(listOf(big, big, big)).vector, 0f)
    }

    @Test fun missingModelIsReportedAsNotReady() = runBlocking {
        runtime.readyOverride = { false }
        try {
            client.embedQuery("x"); fail("expected an error")
        } catch (e: HostCallException) {
            assertEquals(OndolSaem.Errors.MODEL_NOT_READY, e.code)
        }
        assertFalse(client.activeModel(Categories.EMBEDDING)!!.ready)
    }

    @Test fun immediateResidencyUnloadsAfterTheCallButLeaseKeepsIt() = runBlocking {
        container.settings.setResidency(Categories.EMBEDDING, Residency.IMMEDIATE)
        client.embedQuery("a")
        withTimeout(2_000) { while (runtime.embedding.isLoaded) delay(20) }
        assertTrue(embedder.closed)
        val lease = client.acquire(Categories.EMBEDDING)
        client.embedQuery("b"); client.embedQuery("c")
        assertTrue("held by the lease", runtime.embedding.isLoaded)
        // the requests' own leases are returned asynchronously; only the client's lease stays
        fun held() = runtime.embedding.state.value!!.holders.filterValues { it > 0 }
        withTimeout(2_000) { while (held() != mapOf(ctx.packageName to 1)) delay(20) }
        lease.close()
        withTimeout(2_000) { while (runtime.embedding.isLoaded) delay(20) }
    }

    @Test fun chatStreamsAndRunsToolsInTheClient() = runBlocking {
        val events = client.chat(
            "conv1", "sys", listOf(HistoryTurn(true, "earlier"), HistoryTurn(false, "answer")), "find cats",
            tools = listOf(ToolSpec("search", "searches", """{"type":"object","properties":{"q":{"type":"string"}}}""")),
        ) { name, args -> "$name got ${JSONObject(args).getString("q")}" }.toList()
        val ran = events.filterIsInstance<ChatStreamEvent.ToolRan>().single()
        assertEquals("search", ran.name)
        assertEquals("search got find cats", ran.result)
        assertEquals("search got find cats ", events.filterIsInstance<ChatStreamEvent.Token>().joinToString("") { it.text })
        assertEquals("${ctx.packageName}|conv1", chat.lastKey)
        assertEquals(listOf("earlier", "answer"), chat.lastHistory.map { it.text })
    }

    @Test fun chatPassesImages() = runBlocking {
        val text = client.chat("c", "S", emptyList(), "look", userImages = listOf(ByteArray(10), ByteArray(20)))
            .toList().filterIsInstance<ChatStreamEvent.Token>().joinToString("") { it.text }
        assertEquals("images=2 system=S", text)
    }

    @Test fun cancellingTheCollectorCancelsGeneration() = runBlocking {
        val first = client.chat("c", "", emptyList(), "forever").take(1).first()
        assertEquals(ChatStreamEvent.Token("…"), first)
        withTimeout(5_000) { chat.cancelled.await() }
    }

    @Test fun facesComeBackWithBoxesAndVectors() = runBlocking {
        val png = ByteArrayOutputStream().also { Bitmap.createBitmap(40, 30, Bitmap.Config.ARGB_8888).compress(Bitmap.CompressFormat.PNG, 100, it) }.toByteArray()
        val faces = client.detectFaces(png)
        assertEquals(2, faces.size)
        assertEquals(0.1f, faces[1].left, 1e-6f)
        assertArrayEquals(floatArrayOf(1f, 1f), faces[1].vector, 1e-6f)
        assertEquals("face-sface", faces[0].modelId)
        try { client.detectFaces(byteArrayOf(1, 2, 3)); fail() } catch (e: HostCallException) {
            assertEquals(OndolSaem.Errors.BAD_INPUT, e.code)
        }
    }

    @Test fun decisionsAnswerPerHypothesis() = runBlocking {
        val p = client.entail("a note about travel", listOf("This text is about travel.", "This text is about cooking."))
        assertArrayEquals(floatArrayOf(0.9f, 0.1f), p, 1e-6f)
        assertEquals(0, client.entail("x", emptyList()).size)
        assertEquals("nli-mdeberta-v3-w8", client.activeModel(Categories.DECISION)!!.id)
    }

    @Test fun decisionsImageAnswersPerHypothesis() = runBlocking {
        val p = client.entailImage(byteArrayOf(1, 2, 3), "a note about travel", listOf("This text is about travel.", "This text is about cooking."))
        assertArrayEquals(floatArrayOf(0.95f, 0.95f), p, 1e-6f)
    }

    @Test fun imageGenerationReturnsPngBytes() = runBlocking {
        val png = client.generateImage("A cute puppy", 42L)
        assertArrayEquals(byteArrayOf(0x89.toByte(), 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A), png)
    }

    @Test fun decisionsNeedAnApi3Host() = runBlocking {
        container.apiVersion = OndolSaem.Api.V2
        client.disconnect(); delay(200)
        try { client.entail("x", listOf("y")); fail() } catch (e: HostUnavailableException) {
            assertEquals(HostUnavailableException.Reason.HOST_TOO_OLD, e.reason)
        }
    }

    @Test fun facesNeedAnApi2Host() = runBlocking {
        container.apiVersion = OndolSaem.Api.V1 // an older host
        client.disconnect(); delay(200)
        try {
            client.detectFaces(byteArrayOf(1)); fail("expected HOST_TOO_OLD")
        } catch (e: HostUnavailableException) {
            assertEquals(HostUnavailableException.Reason.HOST_TOO_OLD, e.reason)
        }
    }

    @Test fun ocrDecodesTheImage() = runBlocking {
        val bmp = Bitmap.createBitmap(40, 30, Bitmap.Config.ARGB_8888)
        val png = ByteArrayOutputStream().also { bmp.compress(Bitmap.CompressFormat.PNG, 100, it) }.toByteArray()
        assertEquals("40x30", client.recognizeText(png))
        try { client.recognizeText(byteArrayOf(1, 2, 3)); fail() } catch (e: HostCallException) {
            assertEquals(OndolSaem.Errors.BAD_INPUT, e.code)
        }
    }

    @Test fun speechSpansAndTranscriptionWithEarlyStop() = runBlocking {
        val pcm = File.createTempFile("test", ".pcm", ctx.cacheDir).apply { writeBytes(ByteArray(32_000)) }
        try {
            assertTrue(client.canTranscribe())
            assertEquals(listOf(0L..1000L, 5000L..6000L), client.speechSpans(pcm))
            val cues = ArrayList<String>()
            client.transcribe(pcm, 10_000) { start, _, text -> cues += "$start:$text" }
            assertEquals(listOf("10000:cue0", "11000:cue1", "12000:cue2", "13000:cue3", "14000:cue4"), cues)
            // the client stops after two cues: the host must stop too
            val job = async {
                client.transcribe(pcm, 0) { _, _, text -> cues += text; if (text == "cue1") throw kotlinx.coroutines.CancellationException("enough") }
            }
            runCatching { job.await() }
            assertEquals("cue1", cues.last())
        } finally { pcm.delete() }
    }

    @Test fun rawBinderRejectsUnsupportedSpeechLanguage() = runBlocking {
        val pcm = File.createTempFile("invalid-language", ".pcm", ctx.cacheDir).apply { writeBytes(ByteArray(32_000)) }
        try {
            try {
                client.call { host ->
                    val asr = IAsrService.Stub.asInterface(host.getCapability(Categories.ASR))
                    android.os.ParcelFileDescriptor.open(pcm, android.os.ParcelFileDescriptor.MODE_READ_ONLY).use { pfd ->
                        asr.transcribeWithLanguage(pfd, 0, object : ICueCallback.Stub() {
                            override fun onCue(startMs: Long, endMs: Long, text: String) = true
                        }, OndolSaem.PRIORITY_INTERACTIVE, "fr")
                    }
                }
                fail("Unsupported raw Binder language must not reach recognition")
            } catch (e: HostCallException) { assertEquals(OndolSaem.Errors.BAD_INPUT, e.code) }
        } finally { pcm.delete() }
    }

    @Test fun explicitSpeechLanguageDoesNotChangeLegacyAutoAndCanStopEarly() = runBlocking {
        val pcm = File.createTempFile("language", ".pcm", ctx.cacheDir).apply { writeBytes(ByteArray(32_000)) }
        try {
            val heard = ArrayList<String>()
            client.transcribeWithLanguage(pcm, 1500, "en") { start, _, text -> heard += "$start:$text" }
            client.transcribeWithLanguage(pcm, 0, "ko") { _, _, text -> heard += text }
            client.transcribe(pcm, 0) { _, _, text -> heard += text }
            assertEquals(listOf("1500:language:en", "language:ko", "cue0", "cue1", "cue2", "cue3", "cue4"), heard)
            val stopped = ArrayList<String>()
            val job = async { client.transcribeWithLanguage(pcm, 0, "auto") { _, _, text ->
                stopped += text
                if (text == "cue1") throw kotlinx.coroutines.CancellationException("enough")
            } }
            runCatching { job.await() }
            assertEquals(listOf("cue0", "cue1"), stopped)
        } finally { pcm.delete() }
    }

    @Test fun modelChangeReachesConnectedClients() = runBlocking {
        client.apiVersion() // connect and register
        val got = async { withTimeout(5_000) { client.modelChanges.first { it.category == Categories.LLM } } }
        delay(100)
        container.selectModel(Categories.LLM, "qwen35-2b-vl")
        val info = got.await()
        assertEquals("qwen35-2b-vl", info.id)
        assertTrue(info.vision)
        assertNotNull(client.activeModel(Categories.LLM))
        container.settings.setActiveModel(Categories.LLM, "chat-gemma4-e2b")
    }

    @Test fun clientRebindsAfterIdleUnbind() = runBlocking {
        client.apiVersion()
        assertTrue(client.connected.value)
        withTimeout(5_000) { while (client.connected.value) delay(50) }
        assertEquals(OndolSaem.API_VERSION, client.apiVersion())
    }

    // ---- versions --------------------------------------------------------------------------------------------

    @Test fun helloExchangesVersions() = runBlocking {
        val info = client.hostInfo()
        assertEquals(OndolSaem.API_VERSION, info.apiVersion)
        assertEquals(HostContainer.MIN_CLIENT_API, info.minClientApi)
        assertEquals(BuildConfig.VERSION_NAME, info.appVersion)
        assertTrue(client.supports(OndolSaem.Api.V1))
        assertFalse(client.supports(OndolSaem.API_VERSION + 1))
    }

    @Test fun aClientBelowTheHostMinimumIsRefused() = runBlocking {
        container.minClientApi = OndolSaem.API_VERSION + 1
        try {
            client.embedQuery("x"); fail("expected CLIENT_TOO_OLD")
        } catch (e: HostUnavailableException) {
            assertEquals(HostUnavailableException.Reason.CLIENT_TOO_OLD, e.reason)
        }
        assertFalse("not left connected", client.connected.value)
    }

    @Test fun aClientThatNeedsANewerHostSaysSo() = runBlocking {
        val demanding = OndolSaemClient(ctx, idleUnbindMs = 500, minHostApi = OndolSaem.API_VERSION + 1)
        try {
            demanding.apiVersion(); fail("expected HOST_TOO_OLD")
        } catch (e: HostUnavailableException) {
            assertEquals(HostUnavailableException.Reason.HOST_TOO_OLD, e.reason)
        } finally { demanding.disconnect() }
    }

    @Test fun aCallFromANewerLevelIsStoppedBeforeReachingAnOlderHost() = runBlocking {
        val newer = OndolSaem.API_VERSION + 1
        try {
            client.call(minApi = newer) { it.categories() }; fail("expected HOST_TOO_OLD")
        } catch (e: HostUnavailableException) {
            assertEquals(HostUnavailableException.Reason.HOST_TOO_OLD, e.reason)
        }
        // once the host reports that level, the same call goes through
        container.apiVersion = newer
        client.disconnect(); delay(200)
        assertTrue(client.supports(newer))
        assertEquals(ModelCategories.all.size, client.call(minApi = newer) { it.categories() }.size)
    }

    @Test fun anOlderClientLevelIsStillServedWhileAboveTheMinimum() = runBlocking {
        container.apiVersion = OndolSaem.API_VERSION + 3 // a much newer host
        val old = OndolSaemClient(ctx, idleUnbindMs = 500, clientApi = HostContainer.MIN_CLIENT_API)
        try {
            assertEquals("eg2-740m", old.embedQuery("hello").modelId)
        } finally { old.disconnect() }
    }
}
