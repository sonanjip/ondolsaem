package io.github.taehunkim.ondolsaem.runtime

import android.content.Context
import android.os.Debug
import io.github.taehunkim.ondolsaem.ModelInfo
import io.github.taehunkim.ondolsaem.catalog.ModelCatalog
import io.github.taehunkim.ondolsaem.catalog.ModelCategories
import io.github.taehunkim.ondolsaem.catalog.ModelSource
import io.github.taehunkim.ondolsaem.catalog.ModelSpec
import io.github.taehunkim.ondolsaem.download.ModelManager
import io.github.taehunkim.ondolsaem.engine.ChatModel
import io.github.taehunkim.ondolsaem.engine.Decider
import io.github.taehunkim.ondolsaem.engine.NliDecider
import io.github.taehunkim.ondolsaem.engine.LiteRtDecider
import io.github.taehunkim.ondolsaem.engine.Embedder
import io.github.taehunkim.ondolsaem.engine.FaceEngine
import io.github.taehunkim.ondolsaem.engine.SFaceEngine
import io.github.taehunkim.ondolsaem.engine.EmbeddingGemma2Embedder
import io.github.taehunkim.ondolsaem.engine.ImageGenEngine
import io.github.taehunkim.ondolsaem.engine.IrisEngine
import io.github.taehunkim.ondolsaem.engine.LiteRtChatModel
import io.github.taehunkim.ondolsaem.engine.MlKitOcrEngine
import io.github.taehunkim.ondolsaem.engine.NpuTarget
import io.github.taehunkim.ondolsaem.engine.OcrEngine
import io.github.taehunkim.ondolsaem.engine.SherpaSpeech
import io.github.taehunkim.ondolsaem.engine.SpeechTools
import io.github.taehunkim.ondolsaem.sdk.OndolSaem.Categories
import io.github.taehunkim.ondolsaem.settings.HostSettings
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.flow.first
import java.io.File

/** The category has no usable model (not selected, not downloaded, or not implemented). */
class ModelNotReadyException(val category: String, message: String) : Exception(message)

/**
 * Loads each category's active model on demand and unloads it by that category's [Residency]. One resident
 * model per category; changing the active model replaces it.
 */
class ModelRuntime(
    private val context: Context,
    private val manager: ModelManager,
    private val settings: HostSettings,
    scope: CoroutineScope,
) {
    private fun policy(category: String): suspend () -> Residency = { settings.residency(category).first() }

    /** Process PSS in kB; the growth while loading is shown as the model's memory. */
    private val measure: () -> Long? = { runCatching { Debug.getPss() }.getOrNull() }

    val embedding = LeasedResource<Embedder>("embedding", scope, { it.backend.ifEmpty { null } }, measure, policy = policy(Categories.EMBEDDING))
    val llm = LeasedResource<ChatModel>("llm", scope, { it.backend.ifEmpty { null } }, measure, policy = policy(Categories.LLM))
    val ocrRes = LeasedResource<OcrEngine>("ocr", scope, measure = measure, policy = policy(Categories.OCR))
    val asr = LeasedResource<SpeechTools>("asr", scope, measure = measure, policy = policy(Categories.ASR))
    val face = LeasedResource<FaceEngine>("face", scope, measure = measure, policy = policy(Categories.FACE))
    val decision = LeasedResource<Decider>("decision", scope, measure = measure, policy = policy(Categories.DECISION))
    val imageGen = LeasedResource<ImageGenEngine>("image_gen", scope, { it.backend.ifEmpty { null } }, measure, policy = policy(Categories.IMAGE_GEN))

    val resources: Map<String, LeasedResource<*>> = mapOf(
        Categories.EMBEDDING to embedding, Categories.LLM to llm, Categories.OCR to ocrRes, Categories.ASR to asr,
        Categories.FACE to face, Categories.DECISION to decision, Categories.IMAGE_GEN to imageGen,
    )

    /** One request at a time per category, interactive first, background requests round-robin by app. */
    val gates: Map<String, PriorityGate> = resources.keys.associateWith { PriorityGate() }

    private val litertCache get() = File(context.cacheDir, "litertlm")

    /** Test hooks. */
    var embedderFactory: suspend (spec: ModelSpec, file: File, npu: NpuTarget?) -> Embedder? = { spec, file, npu ->
        when (spec.id) {
            "eg2-440m", "eg2-740m" -> EmbeddingGemma2Embedder.load(spec.id, file, litertCache, npu = npu)
            else -> null
        }
    }
    var chatFactory: suspend (spec: ModelSpec, file: File, contextTokens: Int, npu: NpuTarget?) -> ChatModel? = { spec, file, ctx, npu ->
        LiteRtChatModel.load(file, litertCache, spec.vision, ctx, npu, audio = spec.audio)
    }

    /** The NPU build to try first for [spec], when NPU use is on and the build and its libraries are on the phone. */
    private suspend fun npu(spec: ModelSpec): NpuTarget? = if (settings.useNpu.first()) manager.npuTarget(spec) else null
    var ocrFactory: suspend (spec: ModelSpec) -> OcrEngine = { MlKitOcrEngine(it.id) }
    var faceFactory: suspend (spec: ModelSpec, file: File) -> FaceEngine? = { spec, file ->
        when (spec.id) {
            "face-sface" -> SFaceEngine(spec.id, file)
            else -> null
        }
    }
    var deciderFactory: suspend (spec: ModelSpec, dir: File, file: File) -> Decider? = { spec, dir, file ->
        when (spec.id) {
            "nli-mdeberta-v3-w8", "nli-mdeberta-v3-q4" -> NliDecider(spec.id, file, File(dir, "tokenizer.json"))
            "d1-omni-2b" -> LiteRtDecider(spec.id, dir)
            else -> null
        }
    }
    var imageGenFactory: suspend (spec: ModelSpec, dir: File) -> ImageGenEngine? = { spec, dir -> IrisEngine.load(spec.id, dir) }
    var speechFactory: suspend (asrDir: File?) -> SpeechTools = { SherpaSpeech(context, it, 2) }

    suspend fun activeSpec(category: String): ModelSpec? = ModelCatalog.byId(settings.activeModelId(category).first())

    /**
     * Evicts idle heavy models (LLM, Image Gen, and d1-omni-2b decision) before loading another heavy model
     * to prevent out-of-memory crashes on mobile devices.
     */
    suspend fun evictIdleHeavyModels(exceptCategory: String) {
        val heavy = listOf(Categories.LLM, Categories.IMAGE_GEN, Categories.DECISION)
        for (cat in heavy) {
            if (cat == exceptCategory) continue
            if (cat == Categories.DECISION) {
                val decSpec = activeSpec(Categories.DECISION)
                if (decSpec?.id != "d1-omni-2b") continue
            }
            val res = resources[cat]
            if (res != null && res.isLoaded) {
                val trimmed = res.trim()
                if (trimmed) {
                    System.gc()
                }
            }
        }
    }

    /** Test hook: treat models as downloaded. */
    var readyOverride: ((ModelSpec) -> Boolean)? = null

    private fun usable(spec: ModelSpec?): Boolean =
        spec != null && spec.source !is ModelSource.Planned && (readyOverride?.invoke(spec) ?: manager.isReady(spec))

    /** What [category] serves right now (id empty when nothing is selected). */
    suspend fun modelInfo(category: String): ModelInfo {
        val spec = activeSpec(category)
        return ModelInfo().apply {
            this.category = category
            id = spec?.id.orEmpty()
            displayName = spec?.displayName.orEmpty()
            version = spec?.version ?: 0
            ready = usable(spec)
            audio = spec?.audio == true
            vision = spec?.vision == true
            contextTokens = if (category == Categories.LLM) settings.chatContextTokens.first() else 0
        }
    }

    private suspend fun requireSpec(category: String): ModelSpec {
        val spec = activeSpec(category)
        if (!usable(spec)) throw ModelNotReadyException(category, "no ${category} model installed (${spec?.id ?: "none"})")
        return spec!!
    }

    suspend fun embedder(owner: String): Lease<Embedder> {
        val spec = requireSpec(Categories.EMBEDDING)
        val file = manager.file(spec) ?: File("")
        val npu = npu(spec)
        return embedding.acquire(spec.id + if (npu != null) "|npu" else "", owner) { embedderFactory(spec, file, npu) }
            ?: throw ModelNotReadyException(Categories.EMBEDDING, "unsupported embedding model ${spec.id}")
    }

    suspend fun chat(owner: String): Lease<ChatModel> {
        evictIdleHeavyModels(Categories.LLM)
        val spec = requireSpec(Categories.LLM)
        val file = manager.file(spec) ?: File("")
        val ctx = settings.chatContextTokens.first()
        val npu = npu(spec)
        return llm.acquire("${spec.id}|$ctx" + if (npu != null) "|npu" else "", owner) { chatFactory(spec, file, ctx, npu) }
            ?: throw ModelNotReadyException(Categories.LLM, "chat model ${spec.id} did not load")
    }

    suspend fun ocr(owner: String): Lease<OcrEngine> {
        val spec = requireSpec(Categories.OCR)
        return ocrRes.acquire(spec.id, owner) { ocrFactory(spec) }!!
    }

    /** Voice activity detection always works; recognition only when the active ASR model is installed. */
    suspend fun speech(owner: String): Lease<SpeechTools> {
        val spec = activeSpec(Categories.ASR)?.takeIf { usable(it) }
        return asr.acquire(spec?.id ?: "vad-only", owner) { speechFactory(spec?.let(manager::dir)) }!!
    }

    suspend fun faces(owner: String): Lease<FaceEngine> {
        val spec = requireSpec(Categories.FACE)
        val file = manager.file(spec) ?: File("")
        return face.acquire(spec.id, owner) { faceFactory(spec, file) }
            ?: throw ModelNotReadyException(Categories.FACE, "unsupported face model ${spec.id}")
    }

    suspend fun decider(owner: String): Lease<Decider> {
        val spec = requireSpec(Categories.DECISION)
        if (spec.id == "d1-omni-2b") {
            evictIdleHeavyModels(Categories.DECISION)
        }
        val file = manager.modelFile(spec) ?: File("")
        return decision.acquire(spec.id, owner) { deciderFactory(spec, manager.dir(spec), file) }
            ?: throw ModelNotReadyException(Categories.DECISION, "unsupported decision model ${spec.id}")
    }

    suspend fun imageGen(owner: String): Lease<ImageGenEngine> {
        evictIdleHeavyModels(Categories.IMAGE_GEN)
        val spec = requireSpec(Categories.IMAGE_GEN)
        return imageGen.acquire(spec.id, owner) { imageGenFactory(spec, manager.dir(spec)) }
            ?: throw ModelNotReadyException(Categories.IMAGE_GEN, "image generation model ${spec.id} did not load")
    }

    /** A lease on [category]'s model (loads it now). */
    suspend fun lease(category: String, owner: String): Lease<*> = when (category) {
        Categories.EMBEDDING -> embedder(owner)
        Categories.LLM -> chat(owner)
        Categories.OCR -> ocr(owner)
        Categories.ASR -> speech(owner)
        Categories.FACE -> faces(owner)
        Categories.DECISION -> decider(owner)
        Categories.IMAGE_GEN -> imageGen(owner)
        else -> throw ModelNotReadyException(category, "unknown category $category")
    }

    /** The active model changed or was deleted: unload the old one even if in use. */
    suspend fun unload(category: String) { resources[category]?.closeNow() }

    /** Unloads idle models; [includeKept] also those set to stay loaded. Returns how many were unloaded. */
    suspend fun trim(includeKept: Boolean): Int {
        var n = 0
        for ((cat, res) in resources) {
            if (!includeKept && settings.residency(cat).first() == Residency.KEEP) continue
            if (res.trim()) n++
        }
        return n
    }

    suspend fun reapplyPolicies() { resources.values.forEach { it.reapplyPolicy() } }

    /** True while at least one model whose category is set to [Residency.KEEP] is loaded. */
    val keptModelLoaded: Flow<Boolean> = combine(
        ModelCategories.all.map { cat ->
            combine(resources.getValue(cat.id).state, settings.residency(cat.id)) { s, r -> s != null && r == Residency.KEEP }
        },
    ) { flags -> flags.any { it } }
}
