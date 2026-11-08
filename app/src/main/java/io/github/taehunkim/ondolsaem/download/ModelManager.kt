package io.github.taehunkim.ondolsaem.download

import android.content.Context
import androidx.work.Constraints
import androidx.work.ExistingWorkPolicy
import androidx.work.NetworkType
import androidx.work.OneTimeWorkRequestBuilder
import androidx.work.WorkManager
import androidx.work.workDataOf
import io.github.taehunkim.ondolsaem.catalog.ModelCatalog
import io.github.taehunkim.ondolsaem.catalog.ModelSource
import io.github.taehunkim.ondolsaem.catalog.ModelSpec
import io.github.taehunkim.ondolsaem.catalog.NpuDiscovery
import io.github.taehunkim.ondolsaem.catalog.NpuFile
import io.github.taehunkim.ondolsaem.engine.NliW8Converter
import io.github.taehunkim.ondolsaem.engine.Npu
import io.github.taehunkim.ondolsaem.engine.NpuDevice
import io.github.taehunkim.ondolsaem.engine.NpuTarget
import io.github.taehunkim.ondolsaem.settings.HostSettings
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.update
import java.io.File

sealed interface ModelState {
    data object NotInstalled : ModelState
    /** Queued until an unmetered network (Wi-Fi) is available. */
    data object WaitingForWifi : ModelState
    data class Downloading(val downloaded: Long, val total: Long) : ModelState
    /** Downloaded; checking the SHA-256 (and unpacking an archive). */
    data object Verifying : ModelState
    /** Downloaded and verified; being converted on the phone ([ModelSource.HuggingFace.convert]). */
    data class Converting(val done: Long, val total: Long) : ModelState
    data object Installed : ModelState
    data class Failed(val kind: DownloadException.Kind?, val message: String?) : ModelState
    data object Unavailable : ModelState
}

class ModelManager(
    private val context: Context,
    private val settings: HostSettings,
    private val downloader: ModelDownloader = ModelDownloader(),
    private val modelsDir: File = File(context.filesDir, "models"),
    /** This phone's NPU chip (null: no NPU builds apply). */
    val npuDevice: NpuDevice? = Npu.device,
) {
    private val _states = MutableStateFlow<Map<String, ModelState>>(emptyMap())
    val states: StateFlow<Map<String, ModelState>> = _states.asStateFlow()

    /** State of each model's NPU build on this phone (only models with one for [npuDevice]). */
    private val _npuStates = MutableStateFlow<Map<String, ModelState>>(emptyMap())
    val npuStates: StateFlow<Map<String, ModelState>> = _npuStates.asStateFlow()

    val npuRuntime = NpuRuntime(context, downloader)

    /** The last search for NPU builds ([checkNpuBuilds]), [running] while one is under way. */
    data class NpuCheck(val running: Boolean = false, val result: NpuDiscovery.Result? = null, val failed: DownloadException.Kind? = null)

    private val npuCacheFile = File(context.filesDir, "npu-builds.json")
    private val _npuCheck = MutableStateFlow(
        NpuCheck(result = runCatching { NpuDiscovery.load(npuCacheFile) }.getOrNull()?.takeIf { it.soc == npuDevice?.soc }),
    )
    val npuCheck: StateFlow<NpuCheck> = _npuCheck.asStateFlow()

    init { refresh() }

    /** The NPU build of [spec] for this phone, if the last search found one. */
    fun npuSpec(spec: ModelSpec): NpuFile? = _npuCheck.value.result?.builds?.get(spec.id)

    /**
     * Looks up this phone's NPU builds in the models' Hugging Face repositories (at most once a week unless [force]):
     * found builds can be downloaded, the other models show "no NPU build for this chip". Keeps the last result when
     * offline.
     */
    suspend fun checkNpuBuilds(force: Boolean = false, discovery: NpuDiscovery = NpuDiscovery(), now: Long = System.currentTimeMillis()) {
        val device = npuDevice?.takeIf { it.usable } ?: return
        val last = _npuCheck.value
        if (last.running) return
        if (!force && last.result != null && now - last.result.checkedAt < RECHECK_MS && last.result.checked.containsAll(npuCandidates())) return
        _npuCheck.value = last.copy(running = true, failed = null)
        try {
            val r = discovery.check(ModelCatalog.all, device, settings.hfToken.first(), now)
            withContext(Dispatchers.IO) { npuCacheFile.writeText(NpuDiscovery.toJson(r)) }
            _npuCheck.value = NpuCheck(result = r)
        } catch (e: CancellationException) {
            _npuCheck.value = last
            throw e
        } catch (e: Exception) {
            _npuCheck.value = last.copy(running = false, failed = (e as? DownloadException)?.kind ?: DownloadException.Kind.HTTP)
        }
        refresh()
    }

    /** Models that may have NPU builds: LiteRT-LM files on Hugging Face. */
    private fun npuCandidates(): List<String> = ModelCatalog.all.filter { (it.source as? ModelSource.HuggingFace)?.fileName?.endsWith(".litertlm") == true }.map { it.id }

    fun npuFile(spec: ModelSpec): File? = npuSpec(spec)?.let { File(dir(spec), "npu/${it.fileName}") }

    /** The NPU build and its libraries when both are on the phone, else null. */
    fun npuTarget(spec: ModelSpec): NpuTarget? {
        val device = npuDevice ?: return null
        val f = npuFile(spec)?.takeIf { it.exists() } ?: return null
        return npuRuntime.libDir(device)?.let { NpuTarget(f, it) }
    }

    /** Download size of [spec]'s NPU build, plus the Qualcomm runtime when it is still needed. */
    fun npuDownloadBytes(spec: ModelSpec): Long {
        val f = npuSpec(spec) ?: return 0
        return f.sizeBytes + if (npuDevice?.let(npuRuntime::needsInstall) == true) NpuRuntime.QNN_DOWNLOAD_BYTES else 0
    }

    fun file(spec: ModelSpec): File? = when (val s = spec.source) {
        is ModelSource.HuggingFace -> File(modelsDir, "${spec.id}/${s.fileName.substringAfterLast('/')}")
        else -> null
    }

    /** The file the runtime opens: the converted model when the source is converted, else [file]. */
    fun modelFile(spec: ModelSpec): File? = when (val s = spec.source) {
        is ModelSource.HuggingFace -> if (s.convert != null) File(dir(spec), "${s.convert}/model.onnx") else file(spec)
        else -> null
    }

    /** An extra file of a Hugging Face model ([ModelSource.HuggingFace.extras]), stored by its name. */
    fun extra(spec: ModelSpec, path: String): File = File(dir(spec), path.substringAfterLast('/'))

    /** Folder holding the unpacked files of an archive model. */
    fun dir(spec: ModelSpec): File = File(modelsDir, spec.id)

    fun isReady(spec: ModelSpec): Boolean = when (val s = spec.source) {
        is ModelSource.Bundled -> true
        is ModelSource.HuggingFace -> s.extras.keys.all { extra(spec, it).exists() } &&
            (if (s.convert != null) File(dir(spec), "${s.convert}/weights.bin").exists() && modelFile(spec)!!.exists() else file(spec)?.exists() == true)
        is ModelSource.Archive -> s.files.values.all { File(dir(spec), it).exists() }
        is ModelSource.Planned -> false
    }

    fun refresh() {
        val checked = _npuCheck.value.result?.checked.orEmpty()
        _npuStates.update { old ->
            ModelCatalog.all.filter { it.id in checked }.associate { spec ->
                val o = old[spec.id]
                spec.id to when {
                    npuSpec(spec) == null -> ModelState.Unavailable // none published for this chip
                    npuTarget(spec) != null -> ModelState.Installed
                    o is ModelState.Downloading || o is ModelState.WaitingForWifi || o is ModelState.Verifying || o is ModelState.Failed -> o
                    else -> ModelState.NotInstalled
                }
            }
        }
        _states.update { old ->
            ModelCatalog.all.associate { spec ->
                spec.id to when {
                    spec.source is ModelSource.Planned -> ModelState.Unavailable
                    (old[spec.id] is ModelState.Downloading || old[spec.id] is ModelState.WaitingForWifi || old[spec.id] is ModelState.Verifying ||
                        old[spec.id] is ModelState.Converting) &&
                        !isReady(spec) -> old[spec.id]!!
                    isReady(spec) -> ModelState.Installed
                    else -> old[spec.id]?.takeIf { it is ModelState.Failed } ?: ModelState.NotInstalled
                }
            }
        }
    }

    private fun set(id: String, state: ModelState) = _states.update { it + (id to state) }
    private fun setNpu(id: String, state: ModelState) = _npuStates.update { it + (id to state) }

    /** Downloads [id]'s NPU build for this phone (and the Qualcomm runtime when needed). */
    fun startNpuDownload(id: String, wifiOnly: Boolean = false) {
        val spec = ModelCatalog.byId(id) ?: return
        if (npuSpec(spec) == null || npuTarget(spec) != null) return
        setNpu(id, if (wifiOnly) ModelState.WaitingForWifi else ModelState.Downloading(0, npuDownloadBytes(spec)))
        val req = OneTimeWorkRequestBuilder<ModelDownloadWorker>()
            .setInputData(workDataOf(ModelDownloadWorker.KEY_MODEL_ID to id, ModelDownloadWorker.KEY_NPU to true))
            .setConstraints(Constraints.Builder().setRequiredNetworkType(if (wifiOnly) NetworkType.UNMETERED else NetworkType.CONNECTED).build())
            .build()
        WorkManager.getInstance(context).enqueueUniqueWork(npuWorkName(id), ExistingWorkPolicy.REPLACE, req)
    }

    fun cancelNpuDownload(id: String) {
        WorkManager.getInstance(context).cancelUniqueWork(npuWorkName(id))
        setNpu(id, ModelState.NotInstalled)
    }

    /** Deletes [id]'s NPU build; the Qualcomm runtime goes with the last one. */
    fun deleteNpu(id: String) {
        val spec = ModelCatalog.byId(id) ?: return
        WorkManager.getInstance(context).cancelUniqueWork(npuWorkName(id))
        File(dir(spec), "npu").deleteRecursively()
        if (ModelCatalog.all.none { s -> npuFile(s)?.exists() == true }) npuRuntime.delete()
        if (npuSpec(spec) != null) setNpu(id, ModelState.NotInstalled)
    }

    /** Called from [ModelDownloadWorker] for an NPU build. */
    suspend fun runNpuDownload(spec: ModelSpec, onProgress: (Long, Long) -> Unit) {
        val device = npuDevice ?: return
        val f = npuSpec(spec) ?: return
        val target = npuFile(spec)!!
        try {
            if (npuRuntime.needsInstall(device)) {
                npuRuntime.install(device) { d, t -> setNpu(spec.id, ModelState.Downloading(d, t)); onProgress(d, t) }
            }
            if (!target.exists()) {
                val src = spec.source as ModelSource.HuggingFace
                val part = File(target.path + ".dl")
                downloader.download(src.url(f.fileName), part, settings.hfToken.first()) { d, t ->
                    setNpu(spec.id, ModelState.Downloading(d, t)); onProgress(d, t)
                }
                setNpu(spec.id, ModelState.Verifying)
                withContext(Dispatchers.IO) {
                    Checksums.verify(part, f.sha256)
                    if (!part.renameTo(target)) throw java.io.IOException("rename failed")
                }
            }
            setNpu(spec.id, ModelState.Installed)
        } catch (e: CancellationException) {
            setNpu(spec.id, ModelState.NotInstalled)
            throw e
        } catch (e: DownloadException) {
            setNpu(spec.id, ModelState.Failed(e.kind, e.message))
            throw e
        } catch (e: Exception) {
            setNpu(spec.id, ModelState.Failed(null, e.message))
            throw e
        }
    }

    /** [wifiOnly]: wait for an unmetered network (the user declined downloading over a metered one). */
    fun startDownload(id: String, wifiOnly: Boolean = false) {
        val spec = ModelCatalog.byId(id) ?: return
        if (spec.source !is ModelSource.HuggingFace && spec.source !is ModelSource.Archive || isReady(spec)) return
        set(id, if (wifiOnly) ModelState.WaitingForWifi else ModelState.Downloading(0, spec.sizeBytes ?: 0))
        val network = if (wifiOnly) NetworkType.UNMETERED else NetworkType.CONNECTED
        val req = OneTimeWorkRequestBuilder<ModelDownloadWorker>()
            .setInputData(workDataOf(ModelDownloadWorker.KEY_MODEL_ID to id))
            .setConstraints(Constraints.Builder().setRequiredNetworkType(network).build())
            .build()
        // REPLACE: a "wait for Wi-Fi" request may be turned into "download now" and the other way round
        WorkManager.getInstance(context).enqueueUniqueWork(workName(id), ExistingWorkPolicy.REPLACE, req)
    }

    fun cancelDownload(id: String) {
        WorkManager.getInstance(context).cancelUniqueWork(workName(id))
        // partial file is kept so the download can be resumed
        set(id, ModelState.NotInstalled)
    }

    fun delete(id: String) {
        val spec = ModelCatalog.byId(id) ?: return
        WorkManager.getInstance(context).cancelUniqueWork(workName(id))
        file(spec)?.let { f ->
            f.delete()
            File(f.path + ".part").delete()
        }
        val hf = spec.source as? ModelSource.HuggingFace
        if (spec.source is ModelSource.Archive || hf?.extras?.isNotEmpty() == true || hf?.convert != null) {
            dir(spec).deleteRecursively()
        }
        deleteNpu(id) // useless without the main file (the fallback)
        set(id, ModelState.NotInstalled)
    }

    /** Called from [ModelDownloadWorker]. */
    suspend fun runDownload(spec: ModelSpec, onProgress: (Long, Long) -> Unit) {
        val (url, target, token) = when (val src = spec.source) {
            is ModelSource.HuggingFace -> Triple(src.url, file(spec) ?: return, settings.hfToken.first())
            // the token belongs to Hugging Face only
            is ModelSource.Archive -> Triple(src.url, File(dir(spec), ARCHIVE_NAME), "")
            else -> return
        }
        val totalExpected = spec.sizeBytes ?: 0L
        set(spec.id, ModelState.Downloading(0, totalExpected))
        var completedBytes = 0L
        try {
            downloader.download(url, target, token) { d, t ->
                val tot = if (totalExpected > 0) totalExpected else t
                val cur = completedBytes + d
                set(spec.id, ModelState.Downloading(cur, tot))
                onProgress(cur, tot)
            }
            completedBytes += target.length()
            val src0 = spec.source
            if (src0 is ModelSource.HuggingFace) {
                // companion files (chunks, tokenizer, config) with cumulative progress tracking
                for ((path, sha) in src0.extras) {
                    val f = extra(spec, path)
                    if (f.exists()) {
                        completedBytes += f.length()
                        continue
                    }
                    downloader.download(src0.url(path), f, token) { d, _ ->
                        val cur = completedBytes + d
                        val tot = if (totalExpected > 0) totalExpected else cur
                        set(spec.id, ModelState.Downloading(cur, tot))
                        onProgress(cur, tot)
                    }
                    completedBytes += f.length()
                    withContext(Dispatchers.IO) { Checksums.verify(f, sha) }
                }
            }
            set(spec.id, ModelState.Verifying)
            withContext(Dispatchers.IO) {
                when (val src = spec.source) {
                    is ModelSource.HuggingFace -> {
                        Checksums.verify(target, src.sha256)
                        src.convert?.let { conversion ->
                            convert(conversion, spec, target)
                            target.delete() // the converted model replaces the download
                            target.parentFile?.takeIf { it != dir(spec) }?.delete() // its folder, if now empty
                        }
                    }
                    is ModelSource.Archive -> {
                        ArchiveExtractor.extract(target, dir(spec), src.files)
                        target.delete()
                        // a corrupt archive usually fails to unpack; this also catches a wrong but readable one
                        try {
                            src.sha256.forEach { (name, sha) -> Checksums.verify(File(dir(spec), name), sha) }
                        } catch (e: DownloadException) {
                            dir(spec).deleteRecursively()
                            throw e
                        }
                    }
                    else -> Unit
                }
            }
            set(spec.id, ModelState.Installed)
        } catch (e: CancellationException) {
            set(spec.id, ModelState.NotInstalled)
            throw e
        } catch (e: DownloadException) {
            set(spec.id, ModelState.Failed(e.kind, e.message))
            throw e
        } catch (e: Exception) {
            set(spec.id, ModelState.Failed(null, e.message))
            throw e
        }
    }

    /** Converts [source] into `dir/<conversion>/` (template from the APK assets, weights computed here). */
    private fun convert(conversion: String, spec: ModelSpec, source: File) {
        val out = File(dir(spec), conversion).apply { mkdirs() }
        val recipe = context.assets.open("$conversion/recipe.tsv").bufferedReader().use { it.readText() }
        context.assets.open("$conversion/model.onnx").use { i -> File(out, "model.onnx").outputStream().use { i.copyTo(it) } }
        when (conversion) {
            ModelSource.Conversions.NLI_W8 -> NliW8Converter.convert(source, NliW8Converter.parseRecipe(recipe), File(out, "weights.bin")) { d, t ->
                set(spec.id, ModelState.Converting(d, t))
            }
            else -> error("unknown conversion $conversion")
        }
    }

    private fun workName(id: String) = "model-download-$id"
    private fun npuWorkName(id: String) = "model-download-npu-$id"

    private companion object {
        const val ARCHIVE_NAME = "download.tar.bz2"
        const val RECHECK_MS = 7L * 24 * 3600 * 1000
    }
}
