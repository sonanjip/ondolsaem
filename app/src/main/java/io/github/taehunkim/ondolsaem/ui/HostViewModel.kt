package io.github.taehunkim.ondolsaem.ui

import android.app.ActivityManager
import android.app.Application
import android.net.ConnectivityManager
import android.os.Debug
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import io.github.taehunkim.ondolsaem.OndolSaemApp
import io.github.taehunkim.ondolsaem.catalog.ModelCatalog
import io.github.taehunkim.ondolsaem.catalog.ModelCategories
import io.github.taehunkim.ondolsaem.runtime.ResidentState
import io.github.taehunkim.ondolsaem.runtime.Residency
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch

/** Memory numbers for the dashboard (bytes, or kB where noted). */
data class MemorySnapshot(
    val deviceAvail: Long = 0,
    val deviceTotal: Long = 0,
    val deviceThreshold: Long = 0,
    val lowMemory: Boolean = false,
    val processPssKb: Long = 0,
    val nativeKb: Long = 0,
    val graphicsKb: Long = 0,
    val javaKb: Long = 0,
)

class HostViewModel(app: Application) : AndroidViewModel(app) {
    private val c = (app as OndolSaemApp).container
    val settings = c.settings
    val registry = c.registry
    private fun <T> kotlinx.coroutines.flow.Flow<T>.state(initial: T): StateFlow<T> =
        stateIn(viewModelScope, SharingStarted.WhileSubscribed(5_000), initial)

    val modelStates = c.modelManager.states
    val npuStates = c.modelManager.npuStates
    /** This phone's NPU chip, or null when no NPU builds apply. */
    val npuDevice = c.modelManager.npuDevice
    val useNpu: StateFlow<Boolean> = c.settings.useNpu.state(true)
    fun setUseNpu(on: Boolean) {
        viewModelScope.launch {
            c.setUseNpu(on)
            if (on) c.modelManager.checkNpuBuilds()
        }
    }
    val npuCheck = c.modelManager.npuCheck

    /** The NPU self-check: running, and the report so far (null before the first run). */
    data class NpuReport(val running: Boolean, val text: String)
    private val _npuReport = kotlinx.coroutines.flow.MutableStateFlow<NpuReport?>(null)
    val npuReport: StateFlow<NpuReport?> = _npuReport

    fun runNpuSelfCheck() {
        if (_npuReport.value?.running == true) return
        _npuReport.value = NpuReport(true, "")
        viewModelScope.launch(Dispatchers.Default) {
            // free the memory of idle models first: the check loads each build in turn
            c.runtime.trim(includeKept = false)
            val text = try {
                io.github.taehunkim.ondolsaem.runtime.NpuSelfCheck(getApplication(), c.modelManager)
                    .run(c.settings.useNpu.first()) { _npuReport.value = NpuReport(true, it) }
            } catch (t: Throwable) {
                if (t is kotlinx.coroutines.CancellationException) throw t
                (_npuReport.value?.text.orEmpty()) + "\ncheck stopped: $t\n"
            }
            _npuReport.value = NpuReport(false, text)
        }
    }
    fun recheckNpu() { viewModelScope.launch { c.modelManager.checkNpuBuilds(force = true) } }

    init {
        // the user opened the app: look up this chip's NPU builds (at most weekly) while NPU use is on
        viewModelScope.launch { if (c.settings.useNpu.first()) c.modelManager.checkNpuBuilds() }
        viewModelScope.launch { c.modelManager.refreshAiCore() }
    }
    fun npuSize(id: String): Long = ModelCatalog.byId(id)?.let(c.modelManager::npuDownloadBytes) ?: 0
    fun downloadNpu(id: String, wifiOnly: Boolean) = c.modelManager.startNpuDownload(id, wifiOnly)
    fun cancelNpu(id: String) = c.modelManager.cancelNpuDownload(id)
    fun deleteNpu(id: String) { viewModelScope.launch { c.deleteNpu(id) } }
    /** Bytes of the Qualcomm runtime on the phone (0: none needed or not installed). */
    fun npuRuntimeBytes(): Long = npuDevice?.let(c.modelManager.npuRuntime::installedBytes) ?: 0

    val approvals = c.approvals.all
    fun decideApproval(pkg: String, allow: Boolean) = c.decideApproval(pkg, allow)
    fun forgetApproval(pkg: String) = c.approvals.forget(pkg)

    val keepAliveBlocked = c.keepAliveBlocked
    fun retryKeepAlive() = c.retryKeepAlive(force = true)
    fun batteryOptimized(): Boolean =
        !getApplication<Application>().getSystemService(android.os.PowerManager::class.java).isIgnoringBatteryOptimizations(getApplication<Application>().packageName)

    val memory: StateFlow<MemorySnapshot> = flow {
        val am = app.getSystemService(ActivityManager::class.java)
        while (true) {
            val mi = ActivityManager.MemoryInfo().also(am::getMemoryInfo)
            val dm = Debug.MemoryInfo().also(Debug::getMemoryInfo)
            fun stat(k: String) = dm.getMemoryStat(k)?.toLongOrNull() ?: 0
            emit(
                MemorySnapshot(
                    mi.availMem, mi.totalMem, mi.threshold, mi.lowMemory,
                    dm.totalPss.toLong(), stat("summary.native-heap"), stat("summary.graphics"), stat("summary.java-heap"),
                ),
            )
            delay(2_000)
        }
    }.flowOn(Dispatchers.IO).state(MemorySnapshot())

    /** Per category: what is loaded (null = nothing). */
    val resident: StateFlow<Map<String, ResidentState?>> =
        combine(c.runtime.resources.map { (cat, res) -> res.state.map { cat to it } }) { it.toMap() }
            .state(emptyMap())

    val activeIds: StateFlow<Map<String, String>> =
        combine(ModelCategories.all.map { cat -> settings.activeModelId(cat.id).map { cat.id to it } }) { it.toMap() }
            .state(emptyMap())

    val residency: StateFlow<Map<String, Residency>> =
        combine(ModelCategories.all.map { cat -> settings.residency(cat.id).map { cat.id to it } }) { it.toMap() }
            .state(emptyMap())

    val clients: StateFlow<Map<String, Set<String>>> = settings.clients.state(emptyMap())
    val hfToken: StateFlow<String> = settings.hfToken.state("")
    val chatContext: StateFlow<Int> = settings.chatContextTokens.state(0)

    fun isMetered(): Boolean = getApplication<Application>().getSystemService(ConnectivityManager::class.java).isActiveNetworkMetered

    fun download(id: String, wifiOnly: Boolean) = c.modelManager.startDownload(id, wifiOnly)
    fun cancelDownload(id: String) = c.modelManager.cancelDownload(id)
    fun delete(id: String) { viewModelScope.launch { c.deleteModel(id) } }
    fun select(category: String, id: String) { viewModelScope.launch { c.selectModel(category, id) } }

    fun setResidency(category: String, r: Residency) {
        viewModelScope.launch {
            settings.setResidency(category, r)
            c.runtime.resources[category]?.reapplyPolicy()
        }
    }

    fun unload(category: String) { viewModelScope.launch { c.runtime.resources[category]?.trim() } }

    /** Loads the model now; the category's residency decides how long it stays. */
    fun load(category: String) {
        viewModelScope.launch(Dispatchers.Default) {
            runCatching { c.runtime.lease(category, getApplication<Application>().packageName).close() }
        }
    }

    fun setHfToken(t: String) { viewModelScope.launch { settings.setHfToken(t) } }

    fun setChatContext(n: Int) {
        viewModelScope.launch {
            settings.setChatContextTokens(n)
            c.runtime.llm.trim() // reloads with the new size on next use
        }
    }

    fun forgetClient(pkg: String) { viewModelScope.launch { settings.forgetClient(pkg) } }

    fun label(pkg: String) = registry.label(pkg)

    fun spec(id: String?) = id?.let(ModelCatalog::byId)
}
