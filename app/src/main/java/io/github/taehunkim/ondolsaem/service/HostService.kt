package io.github.taehunkim.ondolsaem.service

import android.app.Service
import android.content.Intent
import android.graphics.BitmapFactory
import android.os.IBinder
import android.os.ParcelFileDescriptor
import android.os.SystemClock
import android.util.Log
import io.github.taehunkim.ondolsaem.BuildConfig
import io.github.taehunkim.ondolsaem.ChatRequest
import io.github.taehunkim.ondolsaem.ClientInfo
import io.github.taehunkim.ondolsaem.HostInfo
import io.github.taehunkim.ondolsaem.EmbeddingResult
import io.github.taehunkim.ondolsaem.IAsrService
import io.github.taehunkim.ondolsaem.IChatCallback
import io.github.taehunkim.ondolsaem.IChatService
import io.github.taehunkim.ondolsaem.IClientCallback
import io.github.taehunkim.ondolsaem.ICueCallback
import io.github.taehunkim.ondolsaem.IEmbeddingService
import io.github.taehunkim.ondolsaem.ILease
import io.github.taehunkim.ondolsaem.IOcrService
import io.github.taehunkim.ondolsaem.IFaceService
import io.github.taehunkim.ondolsaem.IDecisionService
import io.github.taehunkim.ondolsaem.IImageGenService
import io.github.taehunkim.ondolsaem.FaceResult
import io.github.taehunkim.ondolsaem.IOndolSaem
import io.github.taehunkim.ondolsaem.IToolExecutor
import io.github.taehunkim.ondolsaem.ModelInfo
import io.github.taehunkim.ondolsaem.OndolSaemApp
import io.github.taehunkim.ondolsaem.catalog.ModelCategories
import io.github.taehunkim.ondolsaem.engine.ChatEvent
import io.github.taehunkim.ondolsaem.engine.ChatTool
import io.github.taehunkim.ondolsaem.engine.ChatTurn
import io.github.taehunkim.ondolsaem.engine.Cue
import io.github.taehunkim.ondolsaem.engine.TurnRole
import io.github.taehunkim.ondolsaem.runtime.Lease
import io.github.taehunkim.ondolsaem.runtime.ModelNotReadyException
import io.github.taehunkim.ondolsaem.sdk.HostErrors
import io.github.taehunkim.ondolsaem.sdk.OndolSaem
import io.github.taehunkim.ondolsaem.sdk.OndolSaem.Categories
import io.github.taehunkim.ondolsaem.sdk.Pfds
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.runInterruptible
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.AtomicBoolean

/** Thrown inside the host for unusable input; reaches the client as [OndolSaem.Errors.BAD_INPUT]. */
class BadInputException(message: String) : Exception(message)

/** The bound service client apps talk to (action [OndolSaem.ACTION_BIND]). */
class HostService : Service() {
    private val container get() = (application as OndolSaemApp).container
    private val runtime get() = container.runtime
    private val caller get() = container.caller
    private val registry get() = container.registry
    private val chatJobs = ConcurrentHashMap<String, Job>()

    override fun onBind(intent: Intent): IBinder? = if (intent.action == OndolSaem.ACTION_BIND) entry else null

    /** Runs a request for [category]: caller check, priority queue, error mapping and usage statistics. */
    private fun <T> serve(category: String, priority: Int, block: suspend (pkg: String) -> T): T {
        val pkg = caller.require()
        container.retryKeepAlive() // succeeds once Android allows it (e.g. battery optimization turned off)
        val start = SystemClock.elapsedRealtime()
        var ok = false
        try {
            val result = runtime.gates.getValue(category).run(priority, pkg) { runBlocking { block(pkg) } }
            ok = true
            return result
        } catch (e: Exception) {
            throw mapError(e)
        } finally {
            registry.record(pkg, category, SystemClock.elapsedRealtime() - start, ok)
        }
    }

    private fun mapError(e: Throwable): RuntimeException = when (e) {
        is SecurityException -> e
        is ModelNotReadyException -> HostErrors.encode(OndolSaem.Errors.MODEL_NOT_READY, e.message)
        is BadInputException -> HostErrors.encode(OndolSaem.Errors.BAD_INPUT, e.message)
        is CancellationException -> HostErrors.encode(OndolSaem.Errors.CANCELLED, e.message)
        else -> {
            Log.w(TAG, "request failed", e)
            val code = if (e.message?.contains("exceeds", ignoreCase = true) == true) OndolSaem.Errors.BAD_INPUT else OndolSaem.Errors.MODEL_FAILED
            HostErrors.encode(code, e.message ?: e.javaClass.simpleName)
        }
    }

    private val entry = object : IOndolSaem.Stub() {
        override fun apiVersion(): Int { caller.require(); return container.apiVersion }

        override fun categories(): Array<ModelInfo> {
            caller.require()
            return runBlocking { ModelCategories.all.map { runtime.modelInfo(it.id) } }.toTypedArray()
        }

        override fun activeModel(category: String): ModelInfo? {
            val pkg = caller.require()
            registry.touch(pkg, category)
            if (ModelCategories.byId(category) == null) return null
            return runBlocking { runtime.modelInfo(category) }
        }

        override fun getCapability(category: String): IBinder? {
            val pkg = caller.require()
            registry.touch(pkg, category)
            return when (category) {
                Categories.EMBEDDING -> embedding
                Categories.LLM -> chat
                Categories.OCR -> ocr
                Categories.ASR -> asr
                Categories.FACE -> face
                Categories.DECISION -> decision
                Categories.IMAGE_GEN -> imageGen
                else -> null
            }
        }

        override fun hello(client: ClientInfo, callback: IClientCallback): HostInfo {
            val pkg = caller.require()
            registry.hello(pkg, client.apiVersion, client.sdkVersion.orEmpty())
            if (client.apiVersion < container.minClientApi) {
                Log.w(TAG, "$pkg uses client API ${client.apiVersion}, below ${container.minClientApi}")
                throw HostErrors.encode(
                    OndolSaem.Errors.CLIENT_TOO_OLD, "client API ${client.apiVersion} < required ${container.minClientApi}",
                )
            }
            container.notifier.callbacks.register(callback, pkg)
            return HostInfo().apply {
                apiVersion = container.apiVersion
                minClientApi = container.minClientApi
                appVersion = BuildConfig.VERSION_NAME
            }
        }

        override fun unregisterClient(callback: IClientCallback) {
            caller.require()
            container.notifier.callbacks.unregister(callback)
        }

        override fun acquire(category: String, clientToken: IBinder): ILease {
            val pkg = caller.require()
            registry.touch(pkg, category)
            val lease = try { runBlocking { runtime.lease(category, pkg) } } catch (e: Exception) { throw mapError(e) }
            return LeaseStub(lease, clientToken)
        }
    }

    private inner class LeaseStub(private val lease: Lease<*>, private val token: IBinder) : ILease.Stub(), IBinder.DeathRecipient {
        private val released = AtomicBoolean(false)

        init {
            try { token.linkToDeath(this, 0) } catch (e: android.os.RemoteException) { close() }
        }

        override fun release() { caller.require(); close() }
        override fun binderDied() = close()

        private fun close() {
            if (!released.compareAndSet(false, true)) return
            runCatching { token.unlinkToDeath(this, 0) }
            lease.close()
        }
    }

    private fun result(v: FloatArray, modelId: String, version: Int) = EmbeddingResult().apply {
        vector = v; this.modelId = modelId; modelVersion = version
    }

    private val embedding = object : IEmbeddingService.Stub() {
        private fun embed(priority: Int, block: suspend (io.github.taehunkim.ondolsaem.engine.Embedder) -> FloatArray): EmbeddingResult =
            serve(Categories.EMBEDDING, priority) { pkg ->
                val version = runtime.activeSpec(Categories.EMBEDDING)?.version ?: 0
                runtime.embedder(pkg).use { l -> result(block(l.value), l.value.modelId, version) }
            }

        override fun embedQuery(text: String, priority: Int) = embed(priority) { it.embedQuery(text) }
        override fun embedDocument(text: String, title: String?, priority: Int) = embed(priority) { it.embedDocument(text, title) }
        override fun embedImage(jpeg: ParcelFileDescriptor, priority: Int) = embed(priority) { it.embedImage(Pfds.readAll(jpeg)) }
        override fun embedImages(jpegs: Array<ParcelFileDescriptor>, priority: Int) =
            embed(priority) { e -> e.embedImages(jpegs.map(Pfds::readAll)) }

        override fun embedAudio(wav: ParcelFileDescriptor, priority: Int) = embed(priority) { e ->
            if (!e.supportsAudio) throw ModelNotReadyException(Categories.EMBEDDING, "${e.modelId} has no audio encoder")
            e.embedAudio(Pfds.readAll(wav))
        }
    }

    private val ocr = object : IOcrService.Stub() {
        override fun recognize(image: ParcelFileDescriptor, priority: Int): String = serve(Categories.OCR, priority) { pkg ->
            val bytes = Pfds.readAll(image)
            val bmp = BitmapFactory.decodeByteArray(bytes, 0, bytes.size) ?: throw BadInputException("unreadable image")
            try { runtime.ocr(pkg).use { it.value.recognize(bmp) } } finally { bmp.recycle() }
        }
    }

    private val face = object : IFaceService.Stub() {
        override fun detectFaces(image: ParcelFileDescriptor, priority: Int): Array<FaceResult> = serve(Categories.FACE, priority) { pkg ->
            val bytes = Pfds.readAll(image)
            val bmp = BitmapFactory.decodeByteArray(bytes, 0, bytes.size) ?: throw BadInputException("unreadable image")
            val spec = runtime.activeSpec(Categories.FACE)
            try {
                runtime.faces(pkg).use { l ->
                    l.value.detect(bmp).map { f ->
                        FaceResult().apply {
                            left = f.left; top = f.top; right = f.right; bottom = f.bottom; score = f.score
                            vector = f.vector; modelId = l.value.modelId; modelVersion = spec?.version ?: 1
                        }
                    }.toTypedArray()
                }
            } finally { bmp.recycle() }
        }
    }

    private val decision = object : IDecisionService.Stub() {
        override fun entail(premise: String?, hypotheses: Array<String>?, priority: Int): FloatArray =
            serve(Categories.DECISION, priority) { pkg ->
                val hs = hypotheses.orEmpty().toList()
                if (hs.size > MAX_HYPOTHESES) throw BadInputException("at most $MAX_HYPOTHESES hypotheses per call")
                runtime.decider(pkg).use { it.value.entail(premise.orEmpty().take(MAX_PREMISE_CHARS), hs) }
            }

        override fun entailImage(image: ParcelFileDescriptor, premise: String?, hypotheses: Array<String>?, priority: Int): FloatArray =
            serve(Categories.DECISION, priority) { pkg ->
                val hs = hypotheses.orEmpty().toList()
                if (hs.size > MAX_HYPOTHESES) throw BadInputException("at most $MAX_HYPOTHESES hypotheses per call")
                val bytes = Pfds.readAll(image)
                runtime.decider(pkg).use { it.value.entailImage(bytes, premise.orEmpty().take(MAX_PREMISE_CHARS), hs) }
            }
    }

    private val imageGen = object : IImageGenService.Stub() {
        override fun getActiveModel(): ModelInfo? {
            val pkg = caller.require()
            registry.touch(pkg, Categories.IMAGE_GEN)
            return runBlocking { runtime.modelInfo(Categories.IMAGE_GEN) }
        }

        override fun generateImage(prompt: String?, seed: Long, priority: Int): ParcelFileDescriptor =
            serve(Categories.IMAGE_GEN, priority) { pkg ->
                val p = prompt.orEmpty().trim()
                if (p.isEmpty()) throw BadInputException("prompt must not be empty")
                val bytes = runtime.imageGen(pkg).use { l ->
                    l.value.generateImage(p, seed)
                }
                Pfds.ofBytes(bytes)
            }
    }

    /** Copies a PCM descriptor to a temporary file (the recognizer seeks in it). */
    private inline fun <T> withPcm(pcm: ParcelFileDescriptor, block: (File) -> T): T {
        val f = File.createTempFile("pcm", ".raw", cacheDir)
        try {
            Pfds.copyTo(pcm, f)
            return block(f)
        } finally { f.delete() }
    }

    private class StopTranscription : Exception()

    private val asr = object : IAsrService.Stub() {
        override fun canTranscribe(): Boolean {
            caller.require()
            return runBlocking { runtime.modelInfo(Categories.ASR).ready }
        }

        override fun speechSpans(pcm: ParcelFileDescriptor, priority: Int): LongArray = serve(Categories.ASR, priority) { pkg ->
            withPcm(pcm) { f ->
                runtime.speech(pkg).use { l -> l.value.speechSpans(f).flatMap { listOf(it.startMs, it.endMs) }.toLongArray() }
            }
        }

        override fun transcribe(pcm: ParcelFileDescriptor, fromMs: Long, callback: ICueCallback, priority: Int) =
            serve(Categories.ASR, priority) { pkg ->
                withPcm(pcm) { f ->
                    runtime.speech(pkg).use { l ->
                        if (!l.value.canTranscribe) throw ModelNotReadyException(Categories.ASR, "no speech recognition model installed")
                        try {
                            l.value.transcribe(f, fromMs) { cue: Cue ->
                                if (!callback.onCue(cue.startMs, cue.endMs, cue.text)) throw StopTranscription()
                            }
                        } catch (_: StopTranscription) {
                        }
                    }
                }
            }
    }

    private val chat = object : IChatService.Stub() {
        override fun reply(request: ChatRequest, callback: IChatCallback, tools: IToolExecutor?, priority: Int) {
            val pkg = caller.require()
            container.retryKeepAlive()
            registry.touch(pkg, Categories.LLM)
            val key = "$pkg|${request.requestId}"
            val job = container.scope.launch(Dispatchers.IO) {
                val gate = runtime.gates.getValue(Categories.LLM)
                val start = SystemClock.elapsedRealtime()
                var ok = false
                var acquired = false
                try {
                    runInterruptible { gate.acquire(priority, pkg) }
                    acquired = true
                    val images = request.userImages.orEmpty().map(Pfds::readAll)
                    val history = request.historyTexts.orEmpty().indices.map { i ->
                        ChatTurn(if (request.historyRoles[i] == 0) TurnRole.USER else TurnRole.ASSISTANT, request.historyTexts[i])
                    }
                    runtime.chat(pkg).use { l ->
                        l.value.reply(
                            "$pkg|${request.conversationKey}", request.system.orEmpty(), history,
                            ChatTurn(TurnRole.USER, request.userText.orEmpty(), images), parseTools(request.toolsJson, tools),
                        ).collect { ev ->
                            when (ev) {
                                is ChatEvent.Token -> callback.onToken(ev.text)
                                is ChatEvent.ToolRan -> callback.onToolRan(ev.name, ev.args, ev.result)
                            }
                        }
                    }
                    ok = true
                    runCatching { callback.onDone() }
                } catch (e: CancellationException) {
                    runCatching { callback.onError(OndolSaem.Errors.CANCELLED, "cancelled") }
                } catch (e: Throwable) {
                    val mapped = mapError(e)
                    val code = HostErrors.decode(mapped as? IllegalStateException ?: IllegalStateException(""))?.code ?: OndolSaem.Errors.MODEL_FAILED
                    runCatching { callback.onError(code, e.message ?: e.javaClass.simpleName) }
                } finally {
                    if (acquired) gate.release()
                    chatJobs.remove(key)
                    registry.record(pkg, Categories.LLM, SystemClock.elapsedRealtime() - start, ok)
                }
            }
            chatJobs[key] = job
        }

        override fun cancel(requestId: String) {
            val pkg = caller.require()
            chatJobs.remove("$pkg|$requestId")?.cancel()
        }

        override fun reset() {
            caller.require()
            runBlocking { runtime.llm.ifLoaded { it.reset() } }
        }
    }

    /** Tools are executed in the client app through [executor]. */
    private fun parseTools(json: String?, executor: IToolExecutor?): List<ChatTool> {
        if (json.isNullOrBlank() || executor == null) return emptyList()
        val arr = JSONArray(json)
        return (0 until arr.length()).map { i ->
            val o = arr.getJSONObject(i)
            object : ChatTool {
                override val name = o.getString("name")
                override val description = o.optString("description")
                override val parameters = o.optJSONObject("parameters")?.toString() ?: "{\"type\":\"object\",\"properties\":{}}"
                override fun execute(args: JSONObject): String =
                    try { executor.execute(name, args.toString()) } catch (e: Exception) { "Error: the app that asked is not reachable (${e.message})" }
            }
        }
    }

    private companion object {
        const val TAG = "OndolSaem.Service"
        const val MAX_HYPOTHESES = 64
        /** The model reads a few hundred tokens; longer text is cut by the tokenizer anyway. */
        const val MAX_PREMISE_CHARS = 8_000
    }
}
