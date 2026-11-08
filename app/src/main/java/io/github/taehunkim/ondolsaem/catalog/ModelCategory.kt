package io.github.taehunkim.ondolsaem.catalog

import androidx.annotation.StringRes
import io.github.taehunkim.ondolsaem.R
import io.github.taehunkim.ondolsaem.runtime.Residency
import io.github.taehunkim.ondolsaem.sdk.OndolSaem.Categories

/** What changing a category's model means for the apps that use it. */
enum class ChangeImpact {
    /** Stored vectors become incompatible: apps must re-embed everything (strong warning). */
    REINDEX_REQUIRED,

    /** Text derived from the model (OCR, transcripts) may change: apps may redo that work. */
    DERIVED_DATA,

    /** Nothing stored depends on the model. */
    NONE,
}

/**
 * A kind of model the host serves. Categories are a registry (string ids from [Categories]), not an enum, so new
 * ones (face recognition, fast decision models, ...) are one more entry plus their AIDL capability.
 */
data class ModelCategory(
    val id: String,
    @StringRes val titleRes: Int,
    @StringRes val descriptionRes: Int,
    val changeImpact: ChangeImpact,
    val defaultResidency: Residency,
    /** Model selected until the user picks another one. */
    val defaultModelId: String,
)

object ModelCategories {
    val all: List<ModelCategory> = listOf(
        ModelCategory(
            // 740M also embeds audio (voice recordings, video sound); installs from before 0.4.0 keep 440M (HostUpgrade)
            Categories.EMBEDDING, R.string.cat_embedding, R.string.cat_embedding_desc,
            ChangeImpact.REINDEX_REQUIRED, Residency.IDLE_2M, "eg2-740m",
        ),
        ModelCategory(
            Categories.LLM, R.string.cat_llm, R.string.cat_llm_desc,
            ChangeImpact.NONE, Residency.IDLE_2M, "chat-gemma4-e2b",
        ),
        ModelCategory(
            Categories.OCR, R.string.cat_ocr, R.string.cat_ocr_desc,
            ChangeImpact.DERIVED_DATA, Residency.IDLE_2M, "mlkit-ko",
        ),
        ModelCategory(
            Categories.ASR, R.string.cat_asr, R.string.cat_asr_desc,
            ChangeImpact.DERIVED_DATA, Residency.IDLE_2M, "asr-sensevoice",
        ),
        ModelCategory(
            Categories.FACE, R.string.cat_face, R.string.cat_face_desc,
            ChangeImpact.REINDEX_REQUIRED, Residency.IDLE_2M, "face-sface",
        ),
        ModelCategory(
            // no warning: apps that keep answers (Jogakbo's tags) store the model id and ask again on their own
            Categories.DECISION, R.string.cat_decision, R.string.cat_decision_desc,
            ChangeImpact.NONE, Residency.IDLE_2M, "nli-mdeberta-v3-w8",
        ),
        ModelCategory(
            Categories.IMAGE_GEN, R.string.cat_image_gen, R.string.cat_image_gen_desc,
            ChangeImpact.NONE, Residency.IMMEDIATE, "iris-3b-turbo",
        ),
    )

    fun byId(id: String): ModelCategory? = all.firstOrNull { it.id == id }
}
