package io.github.taehunkim.ondolsaem.engine.iris

import io.github.taehunkim.ondolsaem.engine.EmbedTableQ8Converter
import io.github.taehunkim.ondolsaem.engine.QuantizedEmbedTable
import java.io.Closeable
import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteOrder
import java.nio.ShortBuffer
import java.nio.channels.FileChannel

/**
 * The iris-3b prompt layout and the host-side embedding lookup.
 */
class PromptTokens(assetsDir: File, embedFile: File? = null) : Closeable {

    private val tokenizer = QwenTokenizer(assetsDir)
    private val prefix = readIds(File(assetsDir, "prompt_prefix_ids.txt"))
    private val suffix = readIds(File(assetsDir, "prompt_suffix_ids.txt"))
    private val actualEmbedFile = embedFile?.takeIf { it.exists() }
        ?: File(assetsDir, Q8_EMBEDDING_FILE).takeIf { it.exists() }
        ?: File(assetsDir, EMBEDDING_FILE)
    private val qTable: QuantizedEmbedTable? =
        if (EmbedTableQ8Converter.isQ8File(actualEmbedFile)) QuantizedEmbedTable.open(actualEmbedFile) else null
    private val embeddingFile: RandomAccessFile? = if (qTable == null) RandomAccessFile(actualEmbedFile, "r") else null
    private val embeddings: ShortBuffer? = embeddingFile?.let { raf ->
        raf.channel
            .map(FileChannel.MapMode.READ_ONLY, 0, raf.length())
            .order(ByteOrder.LITTLE_ENDIAN)
            .asShortBuffer()
    }

    /** Token ids and attention mask, both [SEQ_LEN], for one caption. */
    fun encode(caption: String): Tokens {
        val captionIds = tokenizer.encode(caption)
        val budget = WINDOW - suffix.size
        val n = minOf(captionIds.size, budget)
        val ids = IntArray(SEQ_LEN) { PAD_TOKEN_ID }
        prefix.copyInto(ids, 0)
        for (i in 0 until n) {
            ids[prefix.size + i] = captionIds[i]
        }
        suffix.copyInto(ids, prefix.size + n)
        val realCount = prefix.size + n + suffix.size
        return Tokens(ids, realCount, captionIds.size)
    }

    /** inputs_embeds [1, SEQ_LEN, HIDDEN] as fp32, gathered from the .q8 or bf16 table. */
    fun embed(tokens: Tokens): FloatArray {
        val out = FloatArray(SEQ_LEN * HIDDEN)
        qTable?.let { qt ->
            for (position in 0 until SEQ_LEN) {
                qt.readRow(tokens.ids[position], out, position * HIDDEN)
            }
            return out
        }
        val buf = embeddings!!
        for (position in 0 until SEQ_LEN) {
            val base = tokens.ids[position] * HIDDEN
            for (channel in 0 until HIDDEN) {
                out[position * HIDDEN + channel] = Float.fromBits(buf.get(base + channel).toInt() shl 16)
            }
        }
        return out
    }

    override fun close() {
        qTable?.close()
        embeddingFile?.close()
    }

    /** Padded ids, how many are real (attention mask = 1), and the untruncated caption length. */
    class Tokens(val ids: IntArray, val realCount: Int, val captionTokens: Int) {
        fun mask(): IntArray = IntArray(SEQ_LEN) { if (it < realCount) 1 else 0 }
    }

    companion object {
        const val SEQ_LEN = 334
        const val WINDOW = 300
        const val HIDDEN = 2560
        const val PAD_TOKEN_ID = 151643
        const val EMBEDDING_FILE = "embed_bf16.bin"
        const val Q8_EMBEDDING_FILE = "iris_embed_bf16.q8"

        private fun readIds(file: File): IntArray =
            file.readText().trim().split(',').map { it.trim().toInt() }.toIntArray()
    }
}
