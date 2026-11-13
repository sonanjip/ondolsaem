package io.github.taehunkim.ondolsaem.sdk

/** Constants shared by the OndolSaem host and its client apps. */
object OndolSaem {
    /** Package name of the host app. */
    const val PACKAGE = "io.github.taehunkim.ondolsaem"

    /** Intent action of the host's bound service. */
    const val ACTION_BIND = "$PACKAGE.action.BIND"

    /**
     * Sent by the host to every known client package (explicit, via setPackage) when a category's model changes.
     * Extras: [EXTRA_CATEGORY], [EXTRA_MODEL_ID]. Anyone could send this action, so receivers only treat it as a
     * hint to re-check the host ([OndolSaemClient.activeModel]); the SDK's [ModelChangedReceiver] does that.
     */
    const val ACTION_MODEL_CHANGED = "$PACKAGE.action.MODEL_CHANGED"
    const val EXTRA_CATEGORY = "category"
    const val EXTRA_MODEL_ID = "model_id"

    /**
     * API level of this SDK: the contract it was built against. Bump it whenever a method, parcelable field or
     * category is added, and note the level on the new method ([Api]).
     */
    const val API_VERSION = Api.V5

    /** Release of this SDK (build.gradle.kts `version`). */
    val SDK_VERSION: String get() = BuildConfig.SDK_VERSION

    /** API levels, named so call sites say which level a feature needs. */
    object Api {
        /** hello, categories, activeModel, acquire; embedding, chat, OCR and speech services. */
        const val V1 = 1
        /** Face detection and recognition (category [Categories.FACE], IFaceService). */
        const val V2 = 2
        /** Yes/no decisions (category [Categories.DECISION], IDecisionService). */
        const val V3 = 3
        /** Image generation (category [Categories.IMAGE_GEN], IImageGenService) and multimodal decisions (entailImage). */
        const val V4 = 4
        /** Request-scoped recognition language (IAsrService.transcribeWithLanguage). */
        const val V5 = 5
    }

    /** Requests a user is waiting for (chat, search queries) run before [PRIORITY_BACKGROUND] ones (indexing). */
    const val PRIORITY_INTERACTIVE = 0
    const val PRIORITY_BACKGROUND = 1

    /** Model categories (stable string ids). New categories are added here, never renamed. */
    object Categories {
        const val EMBEDDING = "embedding"
        const val LLM = "llm"
        const val OCR = "ocr"
        const val ASR = "asr"
        /** API 2. Face detection + identity vectors. */
        const val FACE = "face"
        /** API 3. Yes/no decisions: does a statement follow from a text (natural language inference). */
        const val DECISION = "decision"
        /** API 4. Text-to-image generation (IImageGenService). */
        const val IMAGE_GEN = "image_gen"
    }

    /** Error codes of [HostCallException] (sent over Binder by [HostErrors]). */
    object Errors {
        /** The category has no downloaded model (or the model lacks the requested capability). */
        const val MODEL_NOT_READY = 1
        /** The model failed to load or to run. */
        const val MODEL_FAILED = 2
        /** Input rejected (too long, unreadable image, ...). */
        const val BAD_INPUT = 3
        /** The request was cancelled. */
        const val CANCELLED = 4
        /** The client's API level is below the host's minimum: the client app must be updated. */
        const val CLIENT_TOO_OLD = 5
        /**
         * This app is not signed with one of the host's own keys and the user has not allowed it (yet). The first
         * call asks the user in OndolSaem; open [approvalIntent] to take them there. Sent inside a
         * `SecurityException`, so clients built before it was added see a plain refusal (`NOT_ALLOWED`).
         */
        const val NOT_APPROVED = 6
    }

    /** Opens OndolSaem's list of apps waiting for approval. */
    const val ACTION_APPROVE = "$PACKAGE.action.APPROVE"

    /** Intent that opens the approval list in OndolSaem, in OndolSaem's own task. */
    fun approvalIntent(): android.content.Intent =
        android.content.Intent(ACTION_APPROVE).setPackage(PACKAGE).addFlags(android.content.Intent.FLAG_ACTIVITY_NEW_TASK) // its own task, not the client's
}
