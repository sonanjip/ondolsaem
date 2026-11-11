package io.github.taehunkim.ondolsaem.catalog

import androidx.annotation.StringRes
import io.github.taehunkim.ondolsaem.R
import io.github.taehunkim.ondolsaem.sdk.OndolSaem.Categories

sealed interface ModelSource {
    /** Downloaded by the user from Hugging Face into app storage. */
    data class HuggingFace(
        val repo: String,
        val fileName: String,
        val sha256: String? = null,
        /** More files of the same repository (path in the repo → SHA-256), stored next to the main file by name. */
        val extras: Map<String, String?> = emptyMap(),
        /**
         * Converted on the phone after the download (the downloaded main file is then deleted), e.g.
         * [Conversions.NLI_W8] or [Conversions.TFLITE_W8]; the runtime opens [ModelManager.modelFile].
         */
        val convert: String? = null,
        /** Per-extra conversions (key matching [extras] → one of [Conversions]), deleted after conversion. */
        val extraConversions: Map<String, String> = emptyMap(),
    ) : ModelSource {
        val url: String get() = url(fileName)
        fun url(path: String): String = "https://huggingface.co/$repo/resolve/main/$path"
    }

    /**
     * One archive (tar.bz2) downloaded and unpacked; [files] maps an entry name suffix to the local file name,
     * [sha256] the local file name to the expected hash of the unpacked file.
     */
    data class Archive(val url: String, val files: Map<String, String>, val sha256: Map<String, String> = emptyMap()) : ModelSource

    object Conversions {
        /** fp32 mDeBERTa → weight-only int8 (engine/NliW8Converter, assets/nli-w8). */
        const val NLI_W8 = "nli-w8"
        /** fp16 FC weights inside `.tflite` → channel-wise int8 (engine/TfliteW8Converter). */
        const val TFLITE_W8 = "tflite-w8"
        /** `.npy` bf16 embedding table → row-quantized int8 `.q8` (engine/EmbedTableQ8Converter). */
        const val EMBED_NPY_Q8 = "embed-npy-q8"
        /** raw bf16 embedding table → row-quantized int8 `.q8` (engine/EmbedTableQ8Converter). */
        const val EMBED_RAW_Q8 = "embed-raw-q8"

        val ALL = setOf(NLI_W8, TFLITE_W8, EMBED_NPY_Q8, EMBED_RAW_Q8)

        fun convertedFileName(rawName: String, conversion: String): String = when (conversion) {
            TFLITE_W8 -> when {
                rawName.endsWith("_fp16fc.tflite") -> rawName.removeSuffix("_fp16fc.tflite") + "_i8w.tflite"
                rawName.endsWith("_f16w_dit.tflite") -> rawName.removeSuffix("_f16w_dit.tflite") + "_i8w_dit.tflite"
                rawName.endsWith("_f16w.tflite") -> rawName.removeSuffix("_f16w.tflite") + "_i8w.tflite"
                else -> rawName.removeSuffix(".tflite") + "_i8w.tflite"
            }
            EMBED_NPY_Q8, EMBED_RAW_Q8 -> rawName.substringBeforeLast('.') + ".q8"
            else -> rawName
        }
    }

    /** Shipped inside the APK (small). */
    data object Bundled : ModelSource

    /** Listed in the catalog but not implemented yet. */
    data object Planned : ModelSource
}

data class ModelSpec(
    val id: String,
    /** A [ModelCategory] id. */
    val category: String,
    val displayName: String,
    @StringRes val descriptionRes: Int,
    val sizeBytes: Long?,
    val license: String,
    val source: ModelSource,
    /** Embedding models only: can embed audio clips (needs the audio encoder). */
    val audio: Boolean = false,
    /** Chat models only: accepts images as input. */
    val vision: Boolean = false,
    /** Bump when the same id starts producing different outputs (clients then redo their derived data). */
    val version: Int = 1,
)

/**
 * Curated models. SHA-256 values are the Hugging Face LFS ids (`/api/models/<repo>/tree/main`); for the SenseVoice
 * archive they are the hashes of the unpacked files of the official release. Re-check sizes and hashes when a model
 * file changes upstream.
 */
object ModelCatalog {
    val all: List<ModelSpec> = listOf(
        ModelSpec(
            id = "eg2-740m", category = Categories.EMBEDDING,
            displayName = "EmbeddingGemma 2 · 740M (+audio)",
            descriptionRes = R.string.model_eg2_740m_desc,
            sizeBytes = 484_622_336L, license = "Apache-2.0",
            source = ModelSource.HuggingFace(
                "litert-community/embeddinggemma-2-740m-litert-lm",
                "embeddinggemma-2-740m.litertlm",
                "e7a8a2204b91e0f96e92960e84a09a89212e1633dcb7575a9bf3378b4df77f4c",
            ),
            audio = true,
        ),
        ModelSpec(
            id = "chat-gemma4-e2b", category = Categories.LLM,
            displayName = "Gemma 4 E2B (chat, images)",
            descriptionRes = R.string.model_chat_gemma4_e2b_desc,
            sizeBytes = 2_588_147_712L, license = "Apache-2.0",
            source = ModelSource.HuggingFace(
                "litert-community/gemma-4-E2B-it-litert-lm", "gemma-4-E2B-it.litertlm",
                "181938105e0eefd105961417e8da75903eacda102c4fce9ce90f50b97139a63c",
            ),
            vision = true,
        ),
        ModelSpec(
            id = "chat-gemma4-e4b", category = Categories.LLM,
            displayName = "Gemma 4 E4B (chat, images)",
            descriptionRes = R.string.model_chat_gemma4_e4b_desc,
            sizeBytes = 3_659_530_240L, license = "Apache-2.0",
            source = ModelSource.HuggingFace(
                "litert-community/gemma-4-E4B-it-litert-lm", "gemma-4-E4B-it.litertlm",
                "0b2a8980ce155fd97673d8e820b4d29d9c7d99b8fa6806f425d969b145bd52e0",
            ),
            vision = true,
        ),
        ModelSpec(
            id = "qwen35-2b-vl", category = Categories.LLM,
            displayName = "Qwen3.5 2B VL (chat, images)",
            descriptionRes = R.string.model_qwen35_desc,
            sizeBytes = 3_152_378_448L, license = "Apache-2.0",
            source = ModelSource.HuggingFace(
                "litert-community/Qwen3.5-2B", "Qwen3.5-2B-VL_int8.litertlm",
                "33d9d9014eb36f6349e41efd272596756380ff4d0bce2290d0b77fb39c487120",
            ),
            vision = true,
        ),
        ModelSpec(
            id = "lfm3-vl-2b", category = Categories.LLM,
            displayName = "LFM3 VL 2B (chat, images)",
            descriptionRes = R.string.model_lfm3_vl_desc,
            sizeBytes = 2_071_611_643L, license = "LFM Open License 1.0",
            source = ModelSource.HuggingFace(
                "litert-community/LFM3-VL-2B-LiteRT", "LFM3-VL-2B_int4_ctx4096.litertlm",
                "64d89b858df4cea27fea4ebcb3bc468e0bc6af2c2cc87db69ac9776332c451e4",
            ),
            vision = true,
        ),
        ModelSpec(
            id = "lfm3-omni-2b", category = Categories.LLM,
            displayName = "LFM3 Omni 2B (chat, images, audio)",
            descriptionRes = R.string.model_lfm3_omni_desc,
            sizeBytes = 3_226_750_192L, license = "LFM Open License 1.0",
            source = ModelSource.HuggingFace(
                "litert-community/LFM3-Omni-2B-LiteRT", "LFM3-Omni-2B_int8.litertlm",
                "959617973a7ddcb3d887fa0d9891d79556a266490d5c432b02a18ffcac3124ed",
            ),
            vision = true,
            audio = true,
        ),
        ModelSpec(
            id = "mlkit-ko", category = Categories.OCR,
            displayName = "ML Kit Text Recognition v2 (Korean + Latin)",
            descriptionRes = R.string.model_mlkit_desc,
            sizeBytes = null, license = "ML Kit Terms",
            source = ModelSource.Bundled,
        ),
        ModelSpec(
            id = "asr-sensevoice", category = Categories.ASR,
            displayName = "SenseVoice (Korean, English, Chinese, Japanese)",
            descriptionRes = R.string.model_sensevoice_desc,
            sizeBytes = 163_002_883L, license = "FunASR Model License",
            source = ModelSource.Archive(
                "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17.tar.bz2",
                mapOf("model.int8.onnx" to "model.int8.onnx", "tokens.txt" to "tokens.txt"),
                mapOf(
                    "model.int8.onnx" to "c71f0ce00bec95b07744e116345e33d8cbbe08cef896382cf907bf4b51a2cd51",
                    "tokens.txt" to "f449eb28dc567533d7fa59be34e2abca8784f771850c78a47fb731a31429a1dc",
                ),
            ),
        ),
        ModelSpec(
            id = "face-sface", category = Categories.FACE,
            displayName = "SFace (face recognition) + ML Kit face detection",
            descriptionRes = R.string.model_sface_desc,
            sizeBytes = 38_696_353L, license = "Apache-2.0 (SFace), ML Kit Terms (detection)",
            source = ModelSource.HuggingFace(
                "opencv/face_recognition_sface", "face_recognition_sface_2021dec.onnx",
                "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
            ),
        ),
        ModelSpec(
            // the published int8 builds give wrong answers on sherpa-onnx's ONNX Runtime (see NliW8Converter):
            // the fp32 file is downloaded and turned into weight-only int8 (341 MB) on the phone
            id = "nli-mdeberta-v3-w8", category = Categories.DECISION,
            displayName = "mDeBERTa-v3 NLI · int8 (multilingual, Korean)",
            descriptionRes = R.string.model_nli_w8_desc,
            sizeBytes = 1_116_156_872L + 16_316_225L, license = "MIT",
            source = ModelSource.HuggingFace(
                NLI_REPO, "onnx/model.onnx", "af43c04cb326eec0629bb8de8ccd4d8a73db6030272047c5f03f66277d059437",
                extras = mapOf("tokenizer.json" to NLI_TOKENIZER_SHA), convert = ModelSource.Conversions.NLI_W8,
            ),
        ),
        ModelSpec(
            id = "nli-mdeberta-v3-q4", category = Categories.DECISION,
            displayName = "mDeBERTa-v3 NLI · q4 (multilingual, Korean)",
            descriptionRes = R.string.model_nli_q4_desc,
            sizeBytes = 886_137_147L + 16_316_225L, license = "MIT",
            source = ModelSource.HuggingFace(
                NLI_REPO, "onnx/model_q4.onnx", "09fcf761615f574942f0a95e08b3221da7e13cb335141609630eaffd223beb17",
                extras = mapOf("tokenizer.json" to NLI_TOKENIZER_SHA),
            ),
        ),
        ModelSpec(
            id = "d1-omni-2b", category = Categories.DECISION,
            displayName = "LiquidAI d1-Omni 2B (text, images)",
            descriptionRes = R.string.model_d1_omni_desc,
            sizeBytes = 2_938_800_160L + 377_915_232L + 524_288_128L + 17_908_634L,
            license = "LFM Open License 1.0",
            source = ModelSource.HuggingFace(
                "litert-community/d1-Omni-2B-LiteRT",
                "d1-omni-2b_decide_L512_fp16fc.tflite",
                "6464159ade002f947f3c67ae02f94bbc0b64ff0d2453b4439b3770211fde9dfd",
                extras = mapOf(
                    "d1-omni-2b_tower_G24x24_fp16fc.tflite" to "b1ea62f74651f4d99c9d26e52d78a698bd05687b81cbd54c4ae9cdad72b6463b",
                    "tables/embed_table.npy" to "0c5cecb5808e459cc5fce45ea85644056294728417e2e2348d447d32e33c27ef",
                    "provider/tokenizer.json" to "41927bd314d17d7d7fadb44e10cd0919f7b71528d7741596d8cbd15c6945ec44",
                ),
                convert = ModelSource.Conversions.TFLITE_W8,
                extraConversions = mapOf(
                    "d1-omni-2b_tower_G24x24_fp16fc.tflite" to ModelSource.Conversions.TFLITE_W8,
                    "tables/embed_table.npy" to ModelSource.Conversions.EMBED_NPY_Q8,
                ),
            ),
            vision = true,
        ),
        ModelSpec(
            id = "iris-3b-turbo", category = Categories.IMAGE_GEN,
            displayName = "Iris 3B Turbo (text-to-image 768px)",
            descriptionRes = R.string.model_iris_desc,
            sizeBytes = 11_308_505_836L,
            license = "Apache-2.0",
            source = ModelSource.HuggingFace(
                "litert-community/iris-3b-Turbo-LiteRT",
                "model/iris3b_config_dit_int8_fp16.json",
                "f9eee22ff08ed94da88a2bc34d9b8ae8467def4514adb4aaaa846f026d32e676",
                extras = mapOf(
                    "model/iris_dit_dual0_i8w_dit.tflite" to "a56b346e7a08aa847ed4765cb7575cb56728e5cc0ac5ceeef5ac0c66b4d23e61",
                    "model/iris_dit_dual1_i8w_dit.tflite" to "48027e1cc2c5a450d89a49f34cc093b737407283cfbb958753d3f3de923a8c10",
                    "model/iris_dit_dual2_i8w_dit.tflite" to "18877cf97bf86fb3168d0339c23a1601c3e8797fdef04015a130dfeb6abdbeb1",
                    "model/iris_dit_single0_i8w_dit.tflite" to "ac6799d48ad92892aaa12ec58c151b22c769b3c14f989d0a8eb2d5e8add0c301",
                    "model/iris_dit_single1_i8w_dit.tflite" to "b2a641e294880c6dfcf43d6758c6445a1819a422c4ef87a92110aadcb2c169e1",
                    "model/iris_dit_single2_i8w_dit.tflite" to "913519074373888342194b84da932690cc346755ea44927a1953e468608dee45",
                    "model/iris_dit_adapter_f16w_dit.tflite" to "8519afb55b512e23168add7a2dbdc606763940b4ab09ff1817810e582252ce10",
                    "model/iris_dit_pixel_f16w_dit.tflite" to "964985151b58d992919d2523fd5628c441332f2019eb971b16c7a444fa408168",
                    "model/iris_host.f32" to "139020066f8cb61ee7116b993c63d9f70957607dcbdd7167107fad76cc55737c",
                    "model/iris_embed_bf16.f32" to "2942e869a0df443edd3158b7f6c4f735f79faea563354b3a077db81f5f21edaa",
                    "model/iris_tokenizer.json" to "dbe0ed41a0b0e0744c1fd307231cfb700003dc164330cc1ebb69fc80e8cd9fa5",
                    "model/iris_te_c00_f16w.tflite" to "f5c8a8c52d045f20b3beb131f47305bd2602848a2f105313980fa0315c3f3350",
                    "model/iris_te_c01_f16w.tflite" to "4a031eba17c4ca7b08ef7111b12f36a266aff21bd685f24d02749c7fd6002226",
                    "model/iris_te_c02_f16w.tflite" to "a4a72795a1ec889f5d717412a43eb12e69703acf099a7899776bf905b69a17b1",
                    "model/iris_te_c03_f16w.tflite" to "34a970242134414210023a36b1dda857818290f9c863352e56fb3f393c4befd4",
                    "model/iris_te_c04_f16w.tflite" to "734f496f22c5ac461503abf9df422e217a5c6387393875e9f49e2aac770f807e",
                    "model/iris_te_c05_f16w.tflite" to "0746bbe9371ba0ff2a44e25c129ce549a8a82613f637935aeee4ecb481a7b853",
                    "model/iris_te_c06_f16w.tflite" to "c9872664b1e90ee87785253258f8c1ab5de8a8e48461836cec1494a5879acf24",
                    "model/iris_te_c07_f16w.tflite" to "ec72b448993bbbe7aa37060a5205936d7e5d109db21ade6421b4203873616cfd",
                    "model/iris_te_c08_f16w.tflite" to "2f612a8ea781716b980a69dfe69e70f961ae9d24dbafb26ac740adcea301127c",
                    "model/iris_te_c09_f16w.tflite" to "8c8acc9b347a13da0c82e2cd6f75da13e01cdff1ef5cfaebbce12286171349a7",
                    "model/iris_te_c10_f16w.tflite" to "45813c0ce1be0e3c2f693f96eb901842d47316610d90a3b18918f250b9e88c58",
                    "model/iris_te_c11_f16w.tflite" to "977236b12c04ce0c10b9ba530db148851756c1c3f263356de103e2216f312277",
                ),
                extraConversions = mapOf(
                    "model/iris_dit_adapter_f16w_dit.tflite" to ModelSource.Conversions.TFLITE_W8,
                    "model/iris_embed_bf16.f32" to ModelSource.Conversions.EMBED_RAW_Q8,
                    "model/iris_te_c00_f16w.tflite" to ModelSource.Conversions.TFLITE_W8,
                    "model/iris_te_c01_f16w.tflite" to ModelSource.Conversions.TFLITE_W8,
                    "model/iris_te_c02_f16w.tflite" to ModelSource.Conversions.TFLITE_W8,
                    "model/iris_te_c03_f16w.tflite" to ModelSource.Conversions.TFLITE_W8,
                    "model/iris_te_c04_f16w.tflite" to ModelSource.Conversions.TFLITE_W8,
                    "model/iris_te_c05_f16w.tflite" to ModelSource.Conversions.TFLITE_W8,
                    "model/iris_te_c06_f16w.tflite" to ModelSource.Conversions.TFLITE_W8,
                    "model/iris_te_c07_f16w.tflite" to ModelSource.Conversions.TFLITE_W8,
                    "model/iris_te_c08_f16w.tflite" to ModelSource.Conversions.TFLITE_W8,
                    "model/iris_te_c09_f16w.tflite" to ModelSource.Conversions.TFLITE_W8,
                    "model/iris_te_c10_f16w.tflite" to ModelSource.Conversions.TFLITE_W8,
                    "model/iris_te_c11_f16w.tflite" to ModelSource.Conversions.TFLITE_W8,
                ),
            ),
        ),
    )

    private const val NLI_REPO = "onnx-community/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7-ONNX"
    private const val NLI_TOKENIZER_SHA = "ac9bd80d04d21b4df95917bdf7c750cbea28acc30a4462c76ca2b9c86d371863"

    fun byId(id: String): ModelSpec? = all.firstOrNull { it.id == id }
    fun ofCategory(category: String): List<ModelSpec> = all.filter { it.category == category }
}
