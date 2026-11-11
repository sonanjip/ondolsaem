package io.github.taehunkim.ondolsaem.engine

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Build
import android.util.Log
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.TensorBuffer
import io.github.taehunkim.ondolsaem.engine.iris.QwenTokenizer
import org.json.JSONObject
import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.ShortBuffer
import java.nio.channels.FileChannel
import java.text.Normalizer
import java.util.regex.Pattern
import kotlin.math.exp
import kotlin.math.max

/**
 * LiteRT-based Decision model (LiquidAI d1-Omni-2B), supporting multimodal decisions
 * (text premise and/or direct image input against multiple hypotheses simultaneously).
 *
 * Uses:
 *  - decide_L512_i8w.tflite (or _fp16fc.tflite fallback): sequence length 512, Set Attention (make_sees)
 *  - tower_G24x24_i8w.tflite (or _fp16fc.tflite fallback): 24x24 patch canvas (384x384 px) -> 144 vision tokens (2048 dims)
 *  - embed_table.q8 (or embed_table.npy fallback): 128,000 x 2048 embedding table accessed via read-only mmap
 *  - tokenizer.json: byte-level BPE tokenizer
 */
class LiteRtDecider(
    override val modelId: String,
    val dir: File,
) : Decider {
    private val decideFile = firstExisting(dir, "d1-omni-2b_decide_L512_i8w.tflite", "d1-omni-2b_decide_L512_fp16fc.tflite")
    private val towerFile = firstExisting(dir, "d1-omni-2b_tower_G24x24_i8w.tflite", "d1-omni-2b_tower_G24x24_fp16fc.tflite")
    private val tableFile = firstExisting(
        dir,
        "embed_table.q8",
        "tables/embed_table.q8",
        "embed_table.npy",
        "tables/embed_table.npy",
    )
    private val tokenizerFile = firstExisting(dir, "tokenizer.json", "provider/tokenizer.json")

    private val tokenizer: D1BpeTokenizer by lazy { D1BpeTokenizer.load(tokenizerFile) }
    private var qTable: QuantizedEmbedTable? = null
    private var tableRaf: RandomAccessFile? = null
    private var embedBuffer: ShortBuffer? = null
    private var tableOffset = 128L

    init {
        initTable()
    }

    private fun initTable() {
        if (!tableFile.exists()) return
        if (EmbedTableQ8Converter.isQ8File(tableFile)) {
            qTable = QuantizedEmbedTable.open(tableFile)
            return
        }
        val raf = RandomAccessFile(tableFile, "r")
        tableRaf = raf
        val ch = raf.channel
        val header = ByteArray(256)
        raf.readFully(header)
        if (header[0] == 0x93.toByte() && header[1] == 'N'.code.toByte()) {
            val hlen = (header[8].toInt() and 0xff) or ((header[9].toInt() and 0xff) shl 8)
            tableOffset = 10L + hlen
        }
        val dataLen = raf.length() - tableOffset
        embedBuffer = ch.map(FileChannel.MapMode.READ_ONLY, tableOffset, dataLen)
            .order(ByteOrder.LITTLE_ENDIAN)
            .asShortBuffer()
    }

    private fun getEmbedRow(tokenId: Int, out: FloatArray, outOffset: Int) {
        qTable?.let { qt ->
            qt.readRow(tokenId, out, outOffset)
            return
        }
        val buf = embedBuffer ?: return
        val base = tokenId.toLong() * HIDDEN
        for (c in 0 until HIDDEN) {
            val bits = buf.get((base + c).toInt()).toInt() and 0xffff
            out[outOffset + c] = Float.fromBits(bits shl 16)
        }
    }

    override fun entail(premise: String, hypotheses: List<String>): FloatArray {
        if (hypotheses.isEmpty()) return FloatArray(0)
        return runInference(imageFeatures = null, premise = premise, hypotheses = hypotheses)
    }

    override fun entailImage(imageBytes: ByteArray, premise: String, hypotheses: List<String>): FloatArray {
        if (hypotheses.isEmpty()) return FloatArray(0)
        val features = extractImageFeatures(imageBytes)
        return runInference(imageFeatures = features, premise = premise, hypotheses = hypotheses)
    }

    override fun entailImage(image: File, premise: String, hypotheses: List<String>): FloatArray {
        return entailImage(image.readBytes(), premise, hypotheses)
    }

    private fun extractImageFeatures(imageBytes: ByteArray): FloatArray? {
        if (!towerFile.exists()) return null
        val bmp = BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size) ?: return null
        val scaled = Bitmap.createScaledBitmap(bmp, CANVAS_PX, CANVAS_PX, true)
        if (scaled != bmp) bmp.recycle()

        // Prepare patches [1, 576, 768] and valid [1, 576, 1]
        val patches = FloatArray(PATCHES_TOTAL * PATCH_FLOATS)
        val valid = FloatArray(PATCHES_TOTAL) { 1.0f }

        // Grid 24x24 patches, each 16x16 px
        val pixels = IntArray(CANVAS_PX * CANVAS_PX)
        scaled.getPixels(pixels, 0, CANVAS_PX, 0, 0, CANVAS_PX, CANVAS_PX)
        scaled.recycle()

        for (py in 0 until GRID_PATCHES) {
            for (px in 0 until GRID_PATCHES) {
                val patchIdx = py * GRID_PATCHES + px
                val patchOffset = patchIdx * PATCH_FLOATS
                var floatIdx = 0
                for (y in 0 until PATCH_SIZE) {
                    for (x in 0 until PATCH_SIZE) {
                        val rgb = pixels[(py * PATCH_SIZE + y) * CANVAS_PX + (px * PATCH_SIZE + x)]
                        val r = ((rgb shr 16) and 0xff) - 127.5f
                        val g = ((rgb shr 8) and 0xff) - 127.5f
                        val b = (rgb and 0xff) - 127.5f
                        patches[patchOffset + floatIdx] = r / 127.5f
                        patches[patchOffset + PATCH_PIXELS + floatIdx] = g / 127.5f
                        patches[patchOffset + 2 * PATCH_PIXELS + floatIdx] = b / 127.5f
                        floatIdx++
                    }
                }
            }
        }

        // Run tower model and close it immediately to save RAM
        val model = createCompiledModel(towerFile) ?: return null
        return try {
            val inPatches = model.createInputBuffer("patches", "tower")
            val inValid = model.createInputBuffer("valid", "tower")
            val outFeat = model.createOutputBuffer("features", "tower")

            inPatches.writeFloat(patches)
            inValid.writeFloat(valid)
            model.run(listOf(inPatches, inValid), listOf(outFeat))

            outFeat.readFloat()
        } finally {
            runCatching { model.close() }
        }
    }

    private fun runInference(imageFeatures: FloatArray?, premise: String, hypotheses: List<String>): FloatArray {
        val out = FloatArray(hypotheses.size)
        val batchHypotheses = hypotheses.take(MAX_HYPOTHESES_PER_PASS)

        // Build prompt tokens
        // Prefix: <|im_start|>user\n [<|vision_start|> [144] <|vision_end|>\n] [premise\n] QUESTION:\n
        val prefixBeforeImage = if (imageFeatures != null) tokenizer.encode("<|im_start|>user\n<|vision_start|>")
            else tokenizer.encode("<|im_start|>user\n" + (if (premise.isNotBlank()) "$premise\n" else "") + "QUESTION:\n")
        val prefixAfterImage = if (imageFeatures != null) tokenizer.encode("<|vision_end|>\n" + (if (premise.isNotBlank()) "$premise\n" else "") + "QUESTION:\n")
            else emptyList()

        val stateLen = if (imageFeatures != null) prefixBeforeImage.size + IMAGE_TOKENS + prefixAfterImage.size else prefixBeforeImage.size
        if (stateLen >= SEQ_LEN - 32) {
            Log.w(TAG, "State length $stateLen exceeds budget, truncating")
            return out
        }

        val questionTags = ArrayList<Int>(SEQ_LEN)
        for (i in 0 until stateLen) questionTags.add(-1)

        val slotPositions = ArrayList<Int>()
        val suffixTokensList = ArrayList<List<Int>>()

        for (k in batchHypotheses.indices) {
            val h = batchHypotheses[k]
            val header = if (k > 0) "QUESTION:\n" else ""
            val suffixText = "$header$h\n\nReply with yes or no only.<|im_end|>\n<|im_start|>assistant\n"
            val tokens = tokenizer.encode(suffixText)
            if (questionTags.size + tokens.size > SEQ_LEN) break
            suffixTokensList.add(tokens)
            for (t in tokens) questionTags.add(k + 1)
            slotPositions.add(questionTags.size - 1)
        }

        val totalLen = questionTags.size
        val (seesQ, seesK) = makeSees(stateLen, questionTags, SEQ_LEN)

        val embeds = FloatArray(SEQ_LEN * HIDDEN)
        var pos = 0
        for (t in prefixBeforeImage) {
            getEmbedRow(t, embeds, pos * HIDDEN)
            pos++
        }
        if (imageFeatures != null) {
            imageFeatures.copyInto(embeds, pos * HIDDEN, 0, IMAGE_TOKENS * HIDDEN)
            pos += IMAGE_TOKENS
            for (t in prefixAfterImage) {
                getEmbedRow(t, embeds, pos * HIDDEN)
                pos++
            }
        }
        for (tokens in suffixTokensList) {
            for (t in tokens) {
                getEmbedRow(t, embeds, pos * HIDDEN)
                pos++
            }
        }

        val model = createCompiledModel(decideFile) ?: return out
        val hidden: FloatArray
        try {
            val inEmbeds = model.createInputBuffer("embeds", "decide")
            val inSeesK = model.createInputBuffer("sees_k", "decide")
            val inSeesQ = model.createInputBuffer("sees_q", "decide")
            val outHidden = model.createOutputBuffer("hidden", "decide")

            inEmbeds.writeFloat(embeds)
            inSeesK.writeFloat(seesK)
            inSeesQ.writeFloat(seesQ)
            model.run(listOf(inEmbeds, inSeesK, inSeesQ), listOf(outHidden))
            hidden = outHidden.readFloat()
        } finally {
            runCatching { model.close() }
        }

        for (k in slotPositions.indices) {
            val slot = slotPositions[k]
            val slotHiddenOffset = slot * HIDDEN

            // Compute logits for YES and NO forms
            var maxYesLogit = -Float.MAX_VALUE
            for (yesToken in YES_TOKENS) {
                val logit = dotProduct(hidden, slotHiddenOffset, yesToken)
                if (logit > maxYesLogit) maxYesLogit = logit
            }

            var maxNoLogit = -Float.MAX_VALUE
            for (noToken in NO_TOKENS) {
                val logit = dotProduct(hidden, slotHiddenOffset, noToken)
                if (logit > maxNoLogit) maxNoLogit = logit
            }

            val sYes = maxYesLogit / TEMPERATURE
            val sNo = maxNoLogit / TEMPERATURE
            val pYes = 1.0f / (1.0f + exp((sNo - sYes).toDouble()).toFloat())
            out[k] = pYes
        }

        return out
    }

    private fun dotProduct(hidden: FloatArray, hiddenOffset: Int, tokenId: Int): Float {
        qTable?.let { return it.dotRow(tokenId, hidden, hiddenOffset) }
        val buf = embedBuffer ?: return 0f
        val base = tokenId.toLong() * HIDDEN
        var sum = 0f
        for (c in 0 until HIDDEN) {
            val bits = buf.get((base + c).toInt()).toInt() and 0xffff
            val w = Float.fromBits(bits shl 16)
            sum += hidden[hiddenOffset + c] * w
        }
        return sum
    }

    private fun makeSees(stateLen: Int, tags: List<Int>, l: Int): Pair<FloatArray, FloatArray> {
        val t = tags.size
        val q = FloatArray(l * SEES)
        val k = FloatArray(l * SEES)

        for (i in 0 until l) {
            q[i * SEES + 1] = 1.0f
        }
        for (i in 0 until t) {
            val isState = i < stateLen
            q[i * SEES + 0] = if (isState) 1.0f else 0.0f
            q[i * SEES + 1] = if (!isState) 1.0f else 0.0f
            k[i * SEES + 0] = 1.0f
            k[i * SEES + 1] = if (isState) 1.0f else 0.0f

            if (!isState) {
                val tag = tags[i]
                if (tag in 1..Q_MAX) {
                    q[i * SEES + 1 + tag] = 1.0f
                    k[i * SEES + 1 + tag] = 1.0f
                }
            }
        }
        return Pair(q, k)
    }

    private fun createCompiledModel(file: File): CompiledModel? {
        if (!file.exists()) return null
        val isEmulator = Build.FINGERPRINT.contains("generic") || Build.HARDWARE.contains("ranchu") || Build.HARDWARE.contains("goldfish")
        return try {
            val accel = if (isEmulator) Accelerator.CPU else Accelerator.GPU
            val options = CompiledModel.Options(accel)
            CompiledModel.create(file.absolutePath, options)
        } catch (e: Exception) {
            Log.w(TAG, "GPU compile failed for ${file.name}, falling back to CPU", e)
            runCatching {
                val options = CompiledModel.Options(Accelerator.CPU)
                CompiledModel.create(file.absolutePath, options)
            }.getOrNull()
        }
    }

    override fun close() {
        runCatching { qTable?.close() }
        qTable = null
        runCatching { tableRaf?.close() }
        tableRaf = null
        embedBuffer = null
    }

    companion object {
        private const val TAG = "LiteRtDecider"

        private fun firstExisting(dir: File, vararg names: String): File =
            names.asSequence().map { File(dir, it) }.firstOrNull { it.exists() } ?: File(dir, names.last())
        const val HIDDEN = 2048
        const val SEQ_LEN = 512
        const val Q_MAX = 64
        const val SEES = 2 + Q_MAX
        const val TEMPERATURE = 3.3829832f
        const val MAX_HYPOTHESES_PER_PASS = 16

        const val CANVAS_PX = 384
        const val PATCH_SIZE = 16
        const val GRID_PATCHES = CANVAS_PX / PATCH_SIZE // 24
        const val PATCHES_TOTAL = GRID_PATCHES * GRID_PATCHES // 576
        const val PATCH_PIXELS = PATCH_SIZE * PATCH_SIZE // 256
        const val PATCH_FLOATS = PATCH_PIXELS * 3 // 768
        const val IMAGE_TOKENS = (GRID_PATCHES / 2) * (GRID_PATCHES / 2) // 144

        val YES_TOKENS = intArrayOf(11683, 12447)
        val NO_TOKENS = intArrayOf(2243, 4547, 19598)
    }
}

/** Lightweight BPE tokenizer for d1-Omni. */
class D1BpeTokenizer private constructor(
    private val tokenToId: Map<String, Int>,
    private val mergeRank: Map<Long, Int>,
    private val addedTokens: List<Pair<String, Int>>,
) {
    private val byteEncoder = QwenTokenizer.buildByteEncoder()
    private val addedByChar = addedTokens.sortedByDescending { it.first.length }.groupBy { it.first[0] }

    fun encode(text: String): List<Int> {
        val out = ArrayList<Int>(text.length / 2 + 1)
        var start = 0
        var i = 0
        while (i < text.length) {
            val hit = addedTokenAt(text, i)
            if (hit != null) {
                if (start < i) encodeSegment(text.substring(start, i), out)
                out.add(hit.second)
                i += hit.first.length
                start = i
            } else i++
        }
        if (start < text.length) encodeSegment(text.substring(start), out)
        return out
    }

    private fun addedTokenAt(text: String, index: Int): Pair<String, Int>? {
        val candidates = addedByChar[text[index]] ?: return null
        return candidates.firstOrNull { text.startsWith(it.first, index) }
    }

    private fun encodeSegment(seg: String, out: MutableList<Int>) {
        val norm = Normalizer.normalize(seg, Normalizer.Form.NFC)
        val matcher = PRE_TOKENIZER.matcher(norm)
        var pos = 0
        while (pos < norm.length) {
            if (!matcher.find(pos)) {
                out.addAll(merge(norm.substring(pos)))
                break
            }
            if (matcher.start() > pos) out.addAll(merge(norm.substring(pos, matcher.start())))
            out.addAll(merge(matcher.group()))
            pos = matcher.end()
        }
    }

    private fun merge(piece: String): List<Int> {
        val symbols = ArrayList<Int>(piece.length * 2)
        for (b in piece.toByteArray(Charsets.UTF_8)) {
            val s = String(Character.toChars(byteEncoder[b.toInt() and 0xff]))
            val id = tokenToId[s] ?: continue
            symbols.add(id)
        }
        while (symbols.size > 1) {
            var bestRank = Int.MAX_VALUE
            var bestIdx = -1
            for (j in 0 until symbols.size - 1) {
                val key = (symbols[j].toLong() shl 32) or symbols[j + 1].toLong()
                val r = mergeRank[key] ?: continue
                if (r < bestRank) { bestRank = r; bestIdx = j }
            }
            if (bestIdx < 0) break
            val mergedStr = tokenToId.entries.firstOrNull { it.value == symbols[bestIdx] }?.key.orEmpty() +
                tokenToId.entries.firstOrNull { it.value == symbols[bestIdx + 1] }?.key.orEmpty()
            val mergedId = tokenToId[mergedStr] ?: break
            symbols[bestIdx] = mergedId
            symbols.removeAt(bestIdx + 1)
        }
        return symbols
    }

    companion object {
        private val PRE_TOKENIZER = Pattern.compile("""'(?:[sSſ]|[tT]|[rR][eE]|[vV][eE]|[mM]|[lL][lL]|[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|[\s]*[\r\n]+|[\s]+(?![\s])|[\s]+""")

        fun load(file: File): D1BpeTokenizer {
            if (!file.exists()) return D1BpeTokenizer(emptyMap(), emptyMap(), emptyList())
            val json = JSONObject(file.readText())
            val modelObj = json.optJSONObject("model") ?: JSONObject()
            val vocabObj = modelObj.optJSONObject("vocab") ?: JSONObject()
            val vocab = HashMap<String, Int>(vocabObj.length())
            for (k in vocabObj.keys()) vocab[k] = vocabObj.getInt(k)

            val mergesArr = modelObj.optJSONArray("merges")
            val merges = HashMap<Long, Int>(mergesArr?.length() ?: 0)
            if (mergesArr != null) {
                for (j in 0 until mergesArr.length()) {
                    val m = mergesArr.get(j)
                    val (left, right) = when (m) {
                        is String -> {
                            val sp = m.indexOf(' ')
                            if (sp > 0) m.substring(0, sp) to m.substring(sp + 1) else null
                        }
                        is org.json.JSONArray -> m.getString(0) to m.getString(1)
                        else -> null
                    } ?: continue
                    val lid = vocab[left] ?: continue
                    val rid = vocab[right] ?: continue
                    merges[(lid.toLong() shl 32) or rid.toLong()] = j
                }
            }

            val addedList = ArrayList<Pair<String, Int>>()
            val addedArr = json.optJSONArray("added_tokens")
            if (addedArr != null) {
                for (j in 0 until addedArr.length()) {
                    val a = addedArr.getJSONObject(j)
                    addedList.add(a.getString("content") to a.getInt("id"))
                }
            }
            return D1BpeTokenizer(vocab, merges, addedList)
        }
    }
}
