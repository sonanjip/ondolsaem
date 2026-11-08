package io.github.taehunkim.ondolsaem.engine

import java.io.File
import kotlin.math.exp

/** Yes/no decisions: does each hypothesis follow from the premise. */
interface Decider : AutoCloseable {
    val modelId: String
    /** Probability (0..1) per hypothesis that it follows from [premise]. */
    fun entail(premise: String, hypotheses: List<String>): FloatArray
    /** Probability (0..1) per hypothesis that it follows from [imageBytes] and optional [premise]. */
    fun entailImage(imageBytes: ByteArray, premise: String = "", hypotheses: List<String>): FloatArray =
        throw UnsupportedOperationException("Image decision is not supported by $modelId")
    fun entailImage(image: File, premise: String = "", hypotheses: List<String>): FloatArray =
        entailImage(image.readBytes(), premise, hypotheses)
    override fun close() {}
}

/**
 * A multilingual NLI cross-encoder (mDeBERTa-v3 xnli, int8 ONNX) on sherpa-onnx's ONNX Runtime ([OrtModel]).
 * All hypotheses of one premise run as one batch; P(yes) is the entailment probability of the three-way softmax
 * (entailment, neutral, contradiction). Leaving neutral out (the usual zero-shot reading) made unrelated topics look
 * likely: on Korean notes against 8 topics it gave 21 false yes at 0.70 against 6 with neutral kept.
 */
class NliDecider(
    override val modelId: String,
    model: File,
    tokenizer: File,
    private val maxTokens: Int = MAX_TOKENS,
) : Decider {
    private val tok = UnigramTokenizer.load(tokenizer)
    private val ort = OrtModel(model, threads = 2)

    override fun entail(premise: String, hypotheses: List<String>): FloatArray {
        if (hypotheses.isEmpty()) return FloatArray(0)
        val out = FloatArray(hypotheses.size)
        // batches of up to BATCH rows, padded to the longest row
        for (from in hypotheses.indices step BATCH) {
            val part = hypotheses.subList(from, minOf(hypotheses.size, from + BATCH))
            val rows = part.map { tok.encodePair(premise, it, maxTokens) }
            val len = rows.maxOf { it.size }
            val ids = LongArray(rows.size * len)
            val mask = LongArray(rows.size * len)
            rows.forEachIndexed { r, row -> row.forEachIndexed { c, id -> ids[r * len + c] = id.toLong(); mask[r * len + c] = 1 } }
            val logits = ort.runLong(linkedMapOf("input_ids" to ids, "attention_mask" to mask), longArrayOf(rows.size.toLong(), len.toLong()))
            for (r in rows.indices) out[from + r] = Nli.yes(logits[r * 3], logits[r * 3 + 1], logits[r * 3 + 2])
        }
        return out
    }

    override fun close() = ort.close()

    companion object {
        /** Premise + hypothesis window: the model allows 512; shorter is faster and decisions rarely need more. */
        const val MAX_TOKENS = 320
        private const val BATCH = 8
    }
}

/** NLI output reading, kept apart from [NliDecider] so it is testable without the native runtime. */
object Nli {
    /** softmax(entailment, neutral, contradiction)[0]. */
    fun yes(entail: Float, neutral: Float, contra: Float): Float {
        val m = maxOf(entail, neutral, contra)
        val a = exp((entail - m).toDouble()); val b = exp((neutral - m).toDouble()); val c = exp((contra - m).toDouble())
        return (a / (a + b + c)).toFloat()
    }
}
