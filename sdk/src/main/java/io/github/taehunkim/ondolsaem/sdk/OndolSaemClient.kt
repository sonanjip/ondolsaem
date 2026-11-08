package io.github.taehunkim.ondolsaem.sdk

import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.ServiceConnection
import android.content.pm.PackageManager
import android.os.DeadObjectException
import android.os.IBinder
import android.util.Log
import io.github.taehunkim.ondolsaem.ChatRequest
import io.github.taehunkim.ondolsaem.ClientInfo
import io.github.taehunkim.ondolsaem.HostInfo
import io.github.taehunkim.ondolsaem.EmbeddingResult
import io.github.taehunkim.ondolsaem.FaceResult
import io.github.taehunkim.ondolsaem.IFaceService
import io.github.taehunkim.ondolsaem.IDecisionService
import io.github.taehunkim.ondolsaem.IImageGenService
import io.github.taehunkim.ondolsaem.IAsrService
import io.github.taehunkim.ondolsaem.IChatCallback
import io.github.taehunkim.ondolsaem.IChatService
import io.github.taehunkim.ondolsaem.IClientCallback
import io.github.taehunkim.ondolsaem.ICueCallback
import io.github.taehunkim.ondolsaem.IEmbeddingService
import io.github.taehunkim.ondolsaem.IOcrService
import io.github.taehunkim.ondolsaem.IOndolSaem
import io.github.taehunkim.ondolsaem.IToolExecutor
import io.github.taehunkim.ondolsaem.ModelInfo
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asSharedFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.buffer
import kotlinx.coroutines.flow.callbackFlow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeoutOrNull
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.util.UUID
import java.util.concurrent.atomic.AtomicInteger
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException

/** A tool the chat model may call; [parameters] is a JSON schema object. */
data class ToolSpec(val name: String, val description: String, val parameters: String)

/** One earlier turn of a conversation. */
data class HistoryTurn(val fromUser: Boolean, val text: String)

sealed interface ChatStreamEvent {
    data class Token(val text: String) : ChatStreamEvent
    data class ToolRan(val name: String, val argsJson: String, val result: String) : ChatStreamEvent
}

/** A model kept loaded by [OndolSaemClient.acquire]; [close] lets the host's residency policy take over again. */
class HostLease internal constructor(private val onClose: () -> Unit) : AutoCloseable {
    private var closed = false
    override fun close() { synchronized(this) { if (closed) return; closed = true }; onClose() }
}

/**
 * Connection to the OndolSaem model host. Binds on first use, rebinds after the host process died (calls are
 * retried once, except chat), and unbinds after [idleUnbindMs] without calls or leases. All calls are suspend
 * functions that run on [Dispatchers.IO]; they throw [HostUnavailableException] or [HostCallException].
 *
 * Versions: on every (re)connection the client sends its API level and SDK release (`hello`) and receives the
 * host's ([hostInfo]). A host below [minHostApi] fails with [HostUnavailableException.Reason.HOST_TOO_OLD]; so does
 * a call that needs a newer level than the host has (each method checks the level that added it). A host that no
 * longer serves this client's level answers [HostUnavailableException.Reason.CLIENT_TOO_OLD].
 */
class OndolSaemClient(
    context: Context,
    private val idleUnbindMs: Long = 60_000,
    private val connectTimeoutMs: Long = 15_000,
    /** Lowest host API level this app works with at all. */
    private val minHostApi: Int = OndolSaem.Api.V1,
    /** The API level reported to the host; only tests set it. */
    private val clientApi: Int = OndolSaem.API_VERSION,
) {
    private val app = context.applicationContext
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    private val lock = Mutex()
    @Volatile private var host: IOndolSaem? = null
    @Volatile private var info: HostInfo? = null
    private var connection: ServiceConnection? = null
    private val active = AtomicInteger(0)
    private var unbindJob: Job? = null

    private val _connected = MutableStateFlow(false)
    /** Whether the client is bound right now (it unbinds when idle, so false is normal). */
    val connected: StateFlow<Boolean> = _connected.asStateFlow()

    private val _modelChanges = MutableSharedFlow<ModelInfo>(extraBufferCapacity = 16)
    /** Model changes reported by the host while this client is bound. */
    val modelChanges: SharedFlow<ModelInfo> = _modelChanges.asSharedFlow()

    /** Identifies this client process to the host (it watches the token to drop leases of a dead client). */
    private val token = android.os.Binder()

    private val callback = object : IClientCallback.Stub() {
        override fun onModelChanged(info: ModelInfo) { _modelChanges.tryEmit(info) }
    }

    fun isInstalled(): Boolean = try {
        app.packageManager.getPackageInfo(OndolSaem.PACKAGE, 0); true
    } catch (_: PackageManager.NameNotFoundException) { false }

    /** Opens the host app (null when it is not installed). */
    fun launchIntent(): Intent? = app.packageManager.getLaunchIntentForPackage(OndolSaem.PACKAGE)

    // ---- entry point -------------------------------------------------------------------------------------------

    suspend fun apiVersion(): Int = call { it.apiVersion() }

    /** The connected host's versions (connects first). */
    suspend fun hostInfo(): HostInfo = call { info ?: error("not connected") }

    /** Whether the host implements API level [api] (see [OndolSaem.Api]). */
    suspend fun supports(api: Int): Boolean = hostInfo().apiVersion >= api
    suspend fun categories(): List<ModelInfo> = call { it.categories().toList() }
    suspend fun activeModel(category: String): ModelInfo? = call { it.activeModel(category) }

    /** Loads [category]'s model now and keeps it loaded until the lease is closed (e.g. for one chat turn). */
    suspend fun acquire(category: String): HostLease {
        begin()
        try {
            val lease = call { it.acquire(category, token) }
            return HostLease {
                scope.launch {
                    runCatching { lease.release() }
                    end()
                }
            }
        } catch (e: Throwable) { end(); throw e }
    }

    // ---- embedding ---------------------------------------------------------------------------------------------

    private fun embedding(h: IOndolSaem) =
        IEmbeddingService.Stub.asInterface(h.getCapability(OndolSaem.Categories.EMBEDDING) ?: throw missing(OndolSaem.Categories.EMBEDDING))

    suspend fun embedQuery(text: String, priority: Int = OndolSaem.PRIORITY_INTERACTIVE): EmbeddingResult =
        call { embedding(it).embedQuery(text, priority) }

    suspend fun embedDocument(text: String, title: String?, priority: Int = OndolSaem.PRIORITY_BACKGROUND): EmbeddingResult =
        call { embedding(it).embedDocument(text, title, priority) }

    suspend fun embedImage(jpeg: ByteArray, priority: Int = OndolSaem.PRIORITY_BACKGROUND): EmbeddingResult =
        call { embedding(it).embedImage(Pfds.ofBytes(jpeg), priority) }

    suspend fun embedImages(jpegs: List<ByteArray>, priority: Int = OndolSaem.PRIORITY_BACKGROUND): EmbeddingResult =
        call { embedding(it).embedImages(jpegs.map(Pfds::ofBytes).toTypedArray(), priority) }

    suspend fun embedAudio(wav: ByteArray, priority: Int = OndolSaem.PRIORITY_BACKGROUND): EmbeddingResult =
        call { embedding(it).embedAudio(Pfds.ofBytes(wav), priority) }

    // ---- OCR ---------------------------------------------------------------------------------------------------

    /** [image] is a JPEG or PNG. */
    suspend fun recognizeText(image: ByteArray, priority: Int = OndolSaem.PRIORITY_BACKGROUND): String =
        call { h ->
            IOcrService.Stub.asInterface(h.getCapability(OndolSaem.Categories.OCR) ?: throw missing(OndolSaem.Categories.OCR))
                .recognize(Pfds.ofBytes(image), priority)
        }

    // ---- faces (API 2) -----------------------------------------------------------------------------------------

    /**
     * Faces in [image] (JPEG or PNG): boxes in 0..1 coordinates, quality and identity vectors (cosine similarity:
     * same person scores high). Fails with [HostUnavailableException.Reason.HOST_TOO_OLD] on an API 1 host.
     */
    suspend fun detectFaces(image: ByteArray, priority: Int = OndolSaem.PRIORITY_BACKGROUND): List<FaceResult> =
        call(minApi = OndolSaem.Api.V2) { h ->
            IFaceService.Stub.asInterface(h.getCapability(OndolSaem.Categories.FACE) ?: throw missing(OndolSaem.Categories.FACE))
                .detectFaces(Pfds.ofBytes(image), priority).toList()
        }

    // ---- decisions (API 3, entailImage in API 4) ---------------------------------------------------------------

    /**
     * For each of [hypotheses], the probability (0..1) that it follows from [premise]. Build hypotheses with
     * [Decisions.hypothesis]. Fails with [HostUnavailableException.Reason.HOST_TOO_OLD] on a host below API 3.
     */
    suspend fun entail(premise: String, hypotheses: List<String>, priority: Int = OndolSaem.PRIORITY_BACKGROUND): FloatArray =
        call(minApi = OndolSaem.Api.V3) { h ->
            IDecisionService.Stub.asInterface(h.getCapability(OndolSaem.Categories.DECISION) ?: throw missing(OndolSaem.Categories.DECISION))
                .entail(premise, hypotheses.toTypedArray(), priority)
        }

    /**
     * For each of [hypotheses], the probability (0..1) that it follows from [image] and optional [premise].
     * Fails with [HostUnavailableException.Reason.HOST_TOO_OLD] on a host below API 4.
     */
    suspend fun entailImage(image: File, premise: String = "", hypotheses: List<String>, priority: Int = OndolSaem.PRIORITY_BACKGROUND): FloatArray =
        call(minApi = OndolSaem.Api.V4) { h ->
            IDecisionService.Stub.asInterface(h.getCapability(OndolSaem.Categories.DECISION) ?: throw missing(OndolSaem.Categories.DECISION))
                .entailImage(Pfds.ofFile(image), premise, hypotheses.toTypedArray(), priority)
        }

    suspend fun entailImage(imageBytes: ByteArray, premise: String = "", hypotheses: List<String>, priority: Int = OndolSaem.PRIORITY_BACKGROUND): FloatArray =
        call(minApi = OndolSaem.Api.V4) { h ->
            IDecisionService.Stub.asInterface(h.getCapability(OndolSaem.Categories.DECISION) ?: throw missing(OndolSaem.Categories.DECISION))
                .entailImage(Pfds.ofBytes(imageBytes), premise, hypotheses.toTypedArray(), priority)
        }

    // ---- image generation (API 4) ------------------------------------------------------------------------------

    /** Generates an image for [prompt] with [seed], returning PNG bytes. */
    suspend fun generateImage(prompt: String, seed: Long = 0L, priority: Int = OndolSaem.PRIORITY_INTERACTIVE): ByteArray =
        call(minApi = OndolSaem.Api.V4) { h ->
            val svc = IImageGenService.Stub.asInterface(h.getCapability(OndolSaem.Categories.IMAGE_GEN) ?: throw missing(OndolSaem.Categories.IMAGE_GEN))
            val pfd = svc.generateImage(prompt, seed, priority)
            Pfds.readAll(pfd)
        }

    // ---- speech ------------------------------------------------------------------------------------------------

    private fun asr(h: IOndolSaem) =
        IAsrService.Stub.asInterface(h.getCapability(OndolSaem.Categories.ASR) ?: throw missing(OndolSaem.Categories.ASR))

    suspend fun canTranscribe(): Boolean = call { asr(it).canTranscribe() }

    /** [pcm]: raw 16 kHz mono 16-bit PCM. Returns speech ranges in milliseconds. */
    suspend fun speechSpans(pcm: File, priority: Int = OndolSaem.PRIORITY_BACKGROUND): List<LongRange> =
        call { h -> asr(h).speechSpans(Pfds.ofFile(pcm), priority).toList().chunked(2) { (a, b) -> a..b } }

    /** Streams recognised utterances of [pcm] from [fromMs] on. Cancelling the caller stops at the next utterance. */
    suspend fun transcribe(
        pcm: File, fromMs: Long, priority: Int = OndolSaem.PRIORITY_BACKGROUND,
        onCue: suspend (startMs: Long, endMs: Long, text: String) -> Unit,
    ) {
        val job = currentCoroutineContext()[Job]
        val cb = object : ICueCallback.Stub() {
            override fun onCue(startMs: Long, endMs: Long, text: String): Boolean {
                if (job?.isActive == false) return false
                // The host calls back while our transcribe() transaction is still open; this runs on that thread.
                return try { runBlocking { onCue(startMs, endMs, text) }; job?.isActive != false } catch (e: Exception) {
                    Log.w(TAG, "onCue failed", e); false
                }
            }
        }
        call { asr(it).transcribe(Pfds.ofFile(pcm), fromMs, cb, priority) }
        currentCoroutineContext().ensureActiveCompat()
    }

    // ---- chat --------------------------------------------------------------------------------------------------

    /**
     * Streams an answer. Tool calls are run through [execute] (on a Binder thread). Cancelling the collector cancels
     * the generation in the host. Not retried when the host dies: the flow fails with [HostUnavailableException].
     */
    fun chat(
        conversationKey: String,
        system: String,
        history: List<HistoryTurn>,
        userText: String,
        userImages: List<ByteArray> = emptyList(),
        tools: List<ToolSpec> = emptyList(),
        priority: Int = OndolSaem.PRIORITY_INTERACTIVE,
        execute: (name: String, argsJson: String) -> String = { _, _ -> "Error: no tools" },
    ): Flow<ChatStreamEvent> = callbackFlow {
        begin()
        val requestId = UUID.randomUUID().toString()
        var service: IChatService? = null
        var finished = false
        var death: IBinder.DeathRecipient? = null
        var hostBinder: IBinder? = null
        try {
            val cb = object : IChatCallback.Stub() {
                override fun onToken(text: String) { trySend(ChatStreamEvent.Token(text)) }
                override fun onToolRan(name: String, argsJson: String, result: String) {
                    trySend(ChatStreamEvent.ToolRan(name, argsJson, result))
                }
                override fun onDone() { finished = true; close() }
                override fun onError(code: Int, message: String?) { finished = true; close(HostCallException(code, message)) }
            }
            val executor = if (tools.isEmpty()) null else object : IToolExecutor.Stub() {
                override fun execute(name: String, argsJson: String): String =
                    try { execute(name, argsJson) } catch (e: Exception) { "Error: ${e.message}" }
            }
            val request = ChatRequest().apply {
                this.requestId = requestId
                this.conversationKey = conversationKey
                this.system = system
                historyRoles = IntArray(history.size) { if (history[it].fromUser) 0 else 1 }
                historyTexts = Array(history.size) { history[it].text }
                this.userText = userText
                this.userImages = userImages.map(Pfds::ofBytes).toTypedArray()
                toolsJson = JSONArray(tools.map {
                    JSONObject().put("name", it.name).put("description", it.description).put("parameters", JSONObject(it.parameters))
                }).toString()
            }
            call(retry = false) { h ->
                val svc = IChatService.Stub.asInterface(h.getCapability(OndolSaem.Categories.LLM) ?: throw missing(OndolSaem.Categories.LLM))
                service = svc
                hostBinder = h.asBinder()
                death = IBinder.DeathRecipient {
                    if (!finished) close(HostUnavailableException(HostUnavailableException.Reason.NO_ANSWER, "host died"))
                }.also { h.asBinder().linkToDeath(it, 0) }
                svc.reply(request, cb, executor, priority)
            }
        } catch (e: Throwable) {
            end(); throw e
        }
        awaitClose {
            death?.let { d -> runCatching { hostBinder?.unlinkToDeath(d, 0) } }
            if (!finished) scope.launch { runCatching { service?.cancel(requestId) } }
            end()
        }
    }.buffer(Channel.UNLIMITED).flowOn(Dispatchers.IO)

    /** Drops the host's cached conversation state (history was edited or deleted). */
    suspend fun resetChat() = call { h ->
        IChatService.Stub.asInterface(h.getCapability(OndolSaem.Categories.LLM) ?: throw missing(OndolSaem.Categories.LLM)).reset()
    }

    // ---- connection --------------------------------------------------------------------------------------------

    /** Unbinds now. Later calls bind again. */
    fun disconnect() { scope.launch { lock.withLock { disconnectLocked() } } }

    private fun missing(category: String) = HostCallException(OndolSaem.Errors.MODEL_NOT_READY, "unknown category $category")

    private fun begin() {
        active.incrementAndGet()
        synchronized(this) { unbindJob?.cancel(); unbindJob = null }
    }

    private fun end() {
        if (active.decrementAndGet() > 0) return
        synchronized(this) {
            unbindJob?.cancel()
            unbindJob = scope.launch {
                delay(idleUnbindMs)
                lock.withLock { if (active.get() == 0) disconnectLocked() }
            }
        }
    }

    /** Runs [block] against a connected host on the IO dispatcher; reconnects and retries once if the host died. */
    suspend fun <T> call(retry: Boolean = true, minApi: Int = OndolSaem.Api.V1, block: (IOndolSaem) -> T): T = withContext(Dispatchers.IO) {
        begin()
        try { attempt(retry, minApi, block) } finally { end() }
    }

    private suspend fun <T> attempt(retry: Boolean, minApi: Int, block: (IOndolSaem) -> T): T {
        var attempt = 0
        while (true) {
            val h = connect()
            val hostApi = info?.apiVersion ?: 0
            if (hostApi < minApi) {
                throw HostUnavailableException(
                    HostUnavailableException.Reason.HOST_TOO_OLD, "needs OndolSaem API $minApi, host has $hostApi",
                )
            }
            try {
                return block(h)
            } catch (e: DeadObjectException) {
                lock.withLock { if (host === h) disconnectLocked() }
                if (!retry || attempt++ > 0) throw HostUnavailableException(HostUnavailableException.Reason.NO_ANSWER, "host died", e)
                Log.w(TAG, "host died, reconnecting")
            } catch (e: IllegalStateException) {
                throw HostErrors.decode(e) ?: e
            } catch (e: SecurityException) {
                throw HostErrors.refusal(e)
            }
        }
    }

    private suspend fun connect(): IOndolSaem = lock.withLock {
        host?.takeIf { it.asBinder().isBinderAlive }?.let { return@withLock it }
        disconnectLocked()
        if (!isInstalled()) throw HostUnavailableException(HostUnavailableException.Reason.NOT_INSTALLED)
        val h = withTimeoutOrNull(connectTimeoutMs) { bind() }
            ?: run { disconnectLocked(); throw HostUnavailableException(HostUnavailableException.Reason.NO_ANSWER, "bind timeout") }
        try {
            val hostApi = h.apiVersion()
            if (hostApi < minHostApi) {
                throw HostUnavailableException(HostUnavailableException.Reason.HOST_TOO_OLD, "OndolSaem API $hostApi, need $minHostApi")
            }
            info = h.hello(ClientInfo().apply { apiVersion = clientApi; sdkVersion = OndolSaem.SDK_VERSION }, callback)
        } catch (e: HostUnavailableException) {
            disconnectLocked(); throw e
        } catch (e: IllegalStateException) {
            disconnectLocked()
            val err = HostErrors.decode(e) ?: throw e
            if (err.code == OndolSaem.Errors.CLIENT_TOO_OLD) {
                throw HostUnavailableException(HostUnavailableException.Reason.CLIENT_TOO_OLD, err.message)
            }
            throw err
        } catch (e: SecurityException) {
            disconnectLocked()
            throw HostErrors.refusal(e)
        } catch (e: DeadObjectException) {
            disconnectLocked()
            throw HostUnavailableException(HostUnavailableException.Reason.NO_ANSWER, "host died", e)
        }
        host = h
        _connected.value = true
        h
    }

    private suspend fun bind(): IOndolSaem = suspendCancellableCoroutine { cont ->
        val conn = object : ServiceConnection {
            override fun onServiceConnected(name: ComponentName?, binder: IBinder) {
                if (cont.isActive) cont.resume(IOndolSaem.Stub.asInterface(binder))
            }
            override fun onServiceDisconnected(name: ComponentName?) {
                // the host process died; Android rebinds by itself, the next call notices the dead binder
                host = null; _connected.value = false
            }
            override fun onBindingDied(name: ComponentName?) {
                host = null; _connected.value = false
                if (cont.isActive) cont.resumeWithException(HostUnavailableException(HostUnavailableException.Reason.NO_ANSWER, "binding died"))
                else {
                    val self = this
                    scope.launch { lock.withLock { if (connection === self) disconnectLocked() } }
                }
            }
            override fun onNullBinding(name: ComponentName?) {
                if (cont.isActive) cont.resumeWithException(HostUnavailableException(HostUnavailableException.Reason.NOT_ALLOWED, "null binding"))
            }
        }
        connection = conn
        val intent = Intent(OndolSaem.ACTION_BIND).setPackage(OndolSaem.PACKAGE)
        val ok = try {
            app.bindService(intent, conn, Context.BIND_AUTO_CREATE or Context.BIND_IMPORTANT)
        } catch (e: SecurityException) {
            false
        }
        if (!ok) {
            runCatching { app.unbindService(conn) }
            connection = null
            if (cont.isActive) cont.resumeWithException(HostUnavailableException(HostUnavailableException.Reason.NOT_INSTALLED, "bindService failed"))
        }
    }

    private fun disconnectLocked() {
        host?.let { h -> runCatching { h.unregisterClient(callback) } }
        host = null
        info = null
        connection?.let { runCatching { app.unbindService(it) } }
        connection = null
        _connected.value = false
    }

    private fun kotlin.coroutines.CoroutineContext.ensureActiveCompat() {
        if (this[Job]?.isActive == false) throw kotlinx.coroutines.CancellationException("cancelled")
    }

    private companion object { const val TAG = "OndolSaem.Client" }
}
