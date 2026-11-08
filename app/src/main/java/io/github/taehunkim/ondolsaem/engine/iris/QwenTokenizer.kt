package io.github.taehunkim.ondolsaem.engine.iris

import java.io.File
import java.text.Normalizer
import java.util.regex.Pattern

/**
 * Byte-level BPE tokenizer of the Qwen3-VL-4B-Instruct text encoder (its `tokenizer.json`).
 */
class QwenTokenizer(assetsDir: File) {

    private val tokens: Array<String>
    private val tokenToId = HashMap<String, Int>(VOCAB_HINT)
    private val mergeRank = HashMap<Long, Int>(MERGE_HINT)
    private val addedByFirstChar: Map<Char, List<Pair<String, Int>>>
    private val byteToUnicode: IntArray = buildByteEncoder()

    init {
        val list = ArrayList<String>(VOCAB_HINT)
        File(assetsDir, "qwen_vocab.txt").forEachLine { list.add(it) }
        tokens = list.toTypedArray()
        tokens.forEachIndexed { id, token -> tokenToId[token] = id }
        var rank = 0
        File(assetsDir, "qwen_merges.txt").forEachLine { line ->
            val space = line.indexOf(' ')
            if (space > 0) {
                val left = tokenToId[line.substring(0, space)]
                val right = tokenToId[line.substring(space + 1)]
                if (left != null && right != null) {
                    mergeRank[pairKey(left, right)] = rank
                }
            }
            rank++
        }
        val added = ArrayList<Pair<String, Int>>()
        val addedFile = File(assetsDir, "qwen_added.txt")
        if (addedFile.exists()) {
            addedFile.forEachLine { line ->
                val tab = line.lastIndexOf('\t')
                if (tab > 0) {
                    added.add(line.substring(0, tab) to line.substring(tab + 1).toInt())
                }
            }
        }
        addedByFirstChar = added.sortedByDescending { it.first.length }.groupBy { it.first[0] }
    }

    /** Tokenizes raw text with no template and no special tokens added (`add_special_tokens=False`). */
    fun encode(text: String): List<Int> {
        val ids = ArrayList<Int>(text.length / 2 + 1)
        var segmentStart = 0
        var i = 0
        while (i < text.length) {
            val hit = addedTokenAt(text, i)
            if (hit == null) {
                i++
                continue
            }
            if (segmentStart < i) {
                encodeSegment(text.substring(segmentStart, i), ids)
            }
            ids.add(hit.second)
            i += hit.first.length
            segmentStart = i
        }
        if (segmentStart < text.length) {
            encodeSegment(text.substring(segmentStart), ids)
        }
        return ids
    }

    /** The longest added token that starts at [index], or null. */
    private fun addedTokenAt(text: String, index: Int): Pair<String, Int>? {
        val candidates = addedByFirstChar[text[index]] ?: return null
        return candidates.firstOrNull { text.startsWith(it.first, index) }
    }

    private fun encodeSegment(segment: String, out: MutableList<Int>) {
        val normalized = Normalizer.normalize(segment, Normalizer.Form.NFC)
        val matcher = PRE_TOKENIZER.matcher(normalized)
        var position = 0
        while (position < normalized.length) {
            if (!matcher.find(position)) {
                out.addAll(merge(normalized.substring(position)))
                break
            }
            if (matcher.start() > position) {
                out.addAll(merge(normalized.substring(position, matcher.start())))
            }
            out.addAll(merge(matcher.group()))
            position = matcher.end()
        }
    }

    /** Byte-level encodes one piece and repeatedly merges the lowest-ranked adjacent pair. */
    private fun merge(piece: String): List<Int> {
        val symbols = ArrayList<Int>(piece.length * 2)
        for (byte in piece.toByteArray(Charsets.UTF_8)) {
            val symbol = String(Character.toChars(byteToUnicode[byte.toInt() and BYTE_MASK]))
            val id = tokenToId[symbol] ?: continue
            symbols.add(id)
        }
        while (symbols.size > 1) {
            var bestRank = Int.MAX_VALUE
            var bestIndex = -1
            for (i in 0 until symbols.size - 1) {
                val rank = mergeRank[pairKey(symbols[i], symbols[i + 1])] ?: continue
                if (rank < bestRank) {
                    bestRank = rank
                    bestIndex = i
                }
            }
            if (bestIndex < 0) {
                break
            }
            val left = tokens[symbols[bestIndex]]
            val right = tokens[symbols[bestIndex + 1]]
            val combined = tokenToId[left + right] ?: break
            symbols[bestIndex] = combined
            symbols.removeAt(bestIndex + 1)
        }
        return symbols
    }

    companion object {
        const val WS = "\t\n\u000B\u000c\r \u0085   \u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u2028\u2029\u202f\u205f\u3000"

        val PRE_TOKENIZER: Pattern = Pattern.compile(
            """'(?:[sSſ]|[tT]|[rR][eE]|[vV][eE]|[mM]|[lL][lL]|[dD])""" +
                """|[^\r\n\p{L}\p{N}]?\p{L}+""" +
                """|\p{N}""" +
                """| ?[^$WS\p{L}\p{N}]+[\r\n]*""" +
                """|[$WS]*[\r\n]+""" +
                """|[$WS]+(?![^$WS])""" +
                """|[$WS]+""")

        const val VOCAB_HINT = 1 shl 18
        const val MERGE_HINT = 1 shl 18
        const val BYTE_MASK = 0xff

        fun pairKey(left: Int, right: Int): Long = (left.toLong() shl 32) or right.toLong()

        /** GPT-2 `bytes_to_unicode`: printable bytes map to themselves, the rest from U+0100 on. */
        fun buildByteEncoder(): IntArray {
            val printable = ('!'.code..'~'.code) + ('¡'.code..'¬'.code) + ('®'.code..'ÿ'.code)
            val encoder = IntArray(256)
            var next = 0
            for (byte in 0 until 256) {
                encoder[byte] = if (byte in printable) byte else 256 + next++
            }
            return encoder
        }
    }
}
