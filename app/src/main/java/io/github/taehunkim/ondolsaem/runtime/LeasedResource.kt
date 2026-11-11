package io.github.taehunkim.ondolsaem.runtime

import android.util.Log
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock

/** A borrowed resource. Closing it returns the resource; the resource itself is NOT closed. */
class Lease<T>(val value: T, private val onClose: () -> Unit) : AutoCloseable {
    private var startClose: (() -> Job)? = null
    private var job: Job? = null
    private var closed = false
    override fun close() {
        synchronized(this) {
            if (closed) return
            closed = true
            val starter = startClose
            if (starter != null) job = starter() else onClose()
        }
    }
    /** Closes the lease if needed and suspends until the owner release has been recorded. */
    suspend fun closeAndWait() {
        close()
        job?.join()
    }
    companion object {
        internal fun <T> withJob(value: T, startClose: () -> Job): Lease<T> =
            Lease(value) {}.also { it.startClose = startClose }
    }
}

/** What a [LeasedResource] holds right now, for the dashboard. */
data class ResidentState(
    /** Model key (model id, plus options such as the context size). */
    val key: String,
    /** Backend label such as "GPU" or "CPU", if known. */
    val backend: String?,
    /** Lease count per owner (client package or "host"). */
    val holders: Map<String, Int>,
    /** When the idle timer will unload the model (epoch ms), or null. */
    val unloadAt: Long?,
    /** Process memory growth measured while loading (kB), or null. */
    val loadedKb: Long?,
    val loadedAt: Long,
)

/**
 * Holds one heavyweight resource (a loaded model) and unloads it according to [policy] once nobody is using it.
 * While at least one [Lease] is outstanding the resource is never closed (except by [closeNow]).
 */
class LeasedResource<T : AutoCloseable>(
    private val name: String,
    private val scope: CoroutineScope,
    private val describe: (T) -> String? = { null },
    private val measure: () -> Long? = { null },
    private val clock: () -> Long = System::currentTimeMillis,
    private val policy: suspend () -> Residency,
) {
    private val mutex = Mutex()
    private var current: T? = null
    private var currentKey: String? = null
    private val holders = HashMap<String, Int>()
    private val pendingReleases = ArrayList<Job>()
    private var timer: Job? = null
    private var unloadAt: Long? = null
    private var loadedKb: Long? = null
    private var loadedAt = 0L

    private val _state = MutableStateFlow<ResidentState?>(null)
    val state: StateFlow<ResidentState?> = _state.asStateFlow()

    /** Returns null when [load] yields null. A different [key] replaces (and closes) the resident resource. */
    suspend fun acquire(key: String, owner: String = "host", load: suspend () -> T?): Lease<T>? = mutex.withLock {
        timer?.cancel(); timer = null; unloadAt = null
        if (currentKey != key) closeLocked()
        val res = current ?: run {
            val before = measure()
            val loaded = load() ?: return@withLock null.also { publish() }
            val after = measure()
            current = loaded; currentKey = key; loadedAt = clock()
            loadedKb = if (before != null && after != null) (after - before).coerceAtLeast(0) else null
            Log.i(TAG, "$name loaded ($key)")
            loaded
        }
        holders[owner] = (holders[owner] ?: 0) + 1
        publish()
        Lease.withJob(res) {
            val j = scope.launch(start = kotlinx.coroutines.CoroutineStart.LAZY) {
                kotlinx.coroutines.withContext(kotlinx.coroutines.NonCancellable) { release(owner) }
            }
            synchronized(pendingReleases) { pendingReleases += j }
            j.invokeOnCompletion { synchronized(pendingReleases) { pendingReleases.remove(j) } }
            j.start()
            j
        }
    }

    private suspend fun awaitPendingReleases() {
        while (true) {
            val jobs = synchronized(pendingReleases) { pendingReleases.toList() }
            if (jobs.isEmpty()) break
            jobs.forEach { it.join() }
        }
    }

    private suspend fun release(owner: String) = mutex.withLock {
        val n = (holders[owner] ?: 0) - 1
        if (n > 0) holders[owner] = n else holders.remove(owner)
        if (holders.isNotEmpty() || current == null) { publish(); return@withLock }
        when (val idle = policy().idleMillis) {
            null -> Unit
            0L -> closeLocked()
            else -> {
                unloadAt = clock() + idle
                timer = scope.launch {
                    delay(idle)
                    mutex.withLock { if (holders.isEmpty()) closeLocked() }
                }
            }
        }
        publish()
    }

    /** Policy changed: re-apply it to an idle resource. */
    suspend fun reapplyPolicy() = mutex.withLock {
        if (current == null || holders.isNotEmpty()) return@withLock
        timer?.cancel(); timer = null; unloadAt = null
        when (val idle = policy().idleMillis) {
            null -> Unit
            0L -> closeLocked()
            else -> {
                unloadAt = clock() + idle
                timer = scope.launch { delay(idle); mutex.withLock { if (holders.isEmpty()) closeLocked() } }
            }
        }
        publish()
    }

    /** Unload now if nobody is using it (memory pressure, or the user asked). */
    suspend fun trim(): Boolean {
        awaitPendingReleases()
        return mutex.withLock { if (holders.isEmpty() && current != null) { closeLocked(); true } else false }
    }

    /** Unload regardless of leases (model deleted/changed). */
    suspend fun closeNow() = mutex.withLock { closeLocked() }

    val isLoaded: Boolean get() = current != null

    /** Runs [block] on the loaded resource, if any, without taking a lease. */
    suspend fun ifLoaded(block: (T) -> Unit) = mutex.withLock { current?.let(block) }

    private fun closeLocked() {
        timer?.cancel(); timer = null; unloadAt = null
        current?.let {
            runCatching { it.close() }
            Log.i(TAG, "$name unloaded")
        }
        current = null; currentKey = null; loadedKb = null
        holders.clear()
        publish()
    }

    private fun publish() {
        val c = current
        _state.value = if (c == null) null else ResidentState(
            currentKey.orEmpty(), runCatching { describe(c) }.getOrNull(), HashMap(holders), unloadAt, loadedKb, loadedAt,
        )
    }

    private companion object { const val TAG = "OndolSaem.Residency" }
}
