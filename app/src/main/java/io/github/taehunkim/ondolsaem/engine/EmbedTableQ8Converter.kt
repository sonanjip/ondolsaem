package io.github.taehunkim.ondolsaem.engine

import java.io.BufferedOutputStream
import java.io.Closeable
import java.io.File
import java.io.FileOutputStream
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import kotlin.math.abs

/**
 * Converts host-side BF16 embedding tables (`.npy` or raw little-endian BF16) on the phone into a compact
 * row-quantized INT8 `.q8` file (`OSQ8` v1):
 *
 *   [0..4)    magic "OSQ8"
 *   [4..8)    version = 1 (int32 LE)
 *   [8..12)   rows (int32 LE)
 *   [12..16)  cols (int32 LE)
 *   [16..16 + rows*4)              float32 LE scale per row (max |w| / 127)
 *   [16 + rows*4..end)             int8 weights, row-major [rows, cols]
 */
object EmbedTableQ8Converter {
    internal const val MAGIC = 0x3851534F // "OSQ8" in little-endian
    internal const val VERSION = 1
    internal const val HEADER_BYTES = 16L
    private const val CHUNK = 4 shl 20

    data class NpyLayout(val dataOffset: Long, val rows: Int, val cols: Int)

    fun parseNpyHeader(file: File): NpyLayout = RandomAccessFile(file, "r").use { raf ->
        val prefix = ByteArray(12)
        raf.readFully(prefix)
        check(
            prefix[0] == 0x93.toByte() &&
                prefix[1] == 'N'.code.toByte() &&
                prefix[2] == 'U'.code.toByte() &&
                prefix[3] == 'M'.code.toByte() &&
                prefix[4] == 'P'.code.toByte() &&
                prefix[5] == 'Y'.code.toByte(),
        ) { "Not a .npy file: ${file.name}" }
        val major = prefix[6].toInt() and 0xff
        val headerPrefix: Int
        val hlen: Int
        if (major == 1) {
            headerPrefix = 10
            hlen = (prefix[8].toInt() and 0xff) or ((prefix[9].toInt() and 0xff) shl 8)
        } else {
            headerPrefix = 12
            hlen = (prefix[8].toInt() and 0xff) or
                ((prefix[9].toInt() and 0xff) shl 8) or
                ((prefix[10].toInt() and 0xff) shl 16) or
                ((prefix[11].toInt() and 0xff) shl 24)
        }
        val dictBytes = ByteArray(hlen)
        raf.seek(headerPrefix.toLong())
        raf.readFully(dictBytes)
        val dict = String(dictBytes, Charsets.US_ASCII)
        val shapeMatch = Regex("""'shape'\s*:\s*\(\s*(\d+)\s*,\s*(\d+)\s*,?\s*\)""").find(dict)
            ?: error("Expected 2D shape in .npy header: $dict")
        val rows = shapeMatch.groupValues[1].toInt()
        val cols = shapeMatch.groupValues[2].toInt()
        val dataOffset = headerPrefix.toLong() + hlen
        check(file.length() - dataOffset == rows.toLong() * cols * 2L) {
            "Unexpected .npy payload size in ${file.name}: ${file.length() - dataOffset} vs ${rows.toLong() * cols * 2L}"
        }
        NpyLayout(dataOffset, rows, cols)
    }

    fun isQ8File(file: File): Boolean {
        if (!file.isFile || file.length() < HEADER_BYTES) return false
        return runCatching {
            RandomAccessFile(file, "r").use { raf ->
                val b = ByteArray(4)
                raf.readFully(b)
                b[0] == 'O'.code.toByte() && b[1] == 'S'.code.toByte() &&
                    b[2] == 'Q'.code.toByte() && b[3] == '8'.code.toByte()
            }
        }.getOrDefault(false)
    }

    fun convertNpy(src: File, dst: File, onProgress: (Long, Long) -> Unit = { _, _ -> }) {
        val layout = parseNpyHeader(src)
        convertBf16Region(src, layout.dataOffset, layout.rows, layout.cols, dst, onProgress)
    }

    fun convertRawBf16(src: File, cols: Int, dst: File, onProgress: (Long, Long) -> Unit = { _, _ -> }) {
        val len = src.length()
        val rowBytes = cols.toLong() * 2L
        check(len > 0 && len % rowBytes == 0L) { "Raw BF16 file size $len is not a multiple of $rowBytes" }
        val rows = (len / rowBytes).toInt()
        convertBf16Region(src, 0L, rows, cols, dst, onProgress)
    }

    internal fun convertBf16Region(
        src: File,
        dataOffset: Long,
        rows: Int,
        cols: Int,
        dst: File,
        onProgress: (Long, Long) -> Unit = { _, _ -> },
    ) {
        dst.parentFile?.mkdirs()
        val part = File(dst.path + ".part")
        val scalesSize = rows.toLong() * 4L
        val weightsSize = rows.toLong() * cols.toLong()
        val expectedSize = HEADER_BYTES + scalesSize + weightsSize
        val scales = FloatArray(rows)

        val rowsPerRead = maxOf(1, CHUNK / (cols * 2))
        val inBuf = ByteBuffer.allocate(rowsPerRead * cols * 2).order(ByteOrder.LITTLE_ENDIAN)
        val rowFloats = FloatArray(cols)
        val outBytes = ByteArray(rowsPerRead * cols)

        RandomAccessFile(src, "r").use { raf ->
            val ch = raf.channel
            BufferedOutputStream(FileOutputStream(part), 1 shl 20).use { out ->
                val header = ByteBuffer.allocate(HEADER_BYTES.toInt()).order(ByteOrder.LITTLE_ENDIAN)
                header.putInt(MAGIC).putInt(VERSION).putInt(rows).putInt(cols)
                out.write(header.array())
                // Reserve space for scales; filled after streaming weights.
                val zeros = ByteArray(minOf(scalesSize, CHUNK.toLong()).toInt())
                var rem = scalesSize
                while (rem > 0) {
                    val n = minOf(rem, zeros.size.toLong()).toInt()
                    out.write(zeros, 0, n)
                    rem -= n
                }

                var r0 = 0
                while (r0 < rows) {
                    val count = minOf(rowsPerRead, rows - r0)
                    val byteLen = count * cols * 2
                    inBuf.clear()
                    inBuf.limit(byteLen)
                    var pos = dataOffset + r0.toLong() * cols * 2L
                    while (inBuf.hasRemaining()) {
                        val n = ch.read(inBuf, pos)
                        check(n > 0) { "Unexpected EOF reading ${src.name}" }
                        pos += n
                    }
                    inBuf.flip()
                    val sb = inBuf.asShortBuffer()
                    for (i in 0 until count) {
                        var maxAbs = 0f
                        for (j in 0 until cols) {
                            val v = Float.fromBits((sb.get().toInt() and 0xffff) shl 16)
                            rowFloats[j] = v
                            val a = abs(v)
                            if (a > maxAbs) maxAbs = a
                        }
                        val sc = if (maxAbs == 0f) 1e-8f else maxAbs / 127f
                        scales[r0 + i] = sc
                        val base = i * cols
                        for (j in 0 until cols) {
                            outBytes[base + j] = q8(rowFloats[j] / sc)
                        }
                    }
                    out.write(outBytes, 0, count * cols)
                    r0 += count
                    onProgress(r0.toLong() * cols, weightsSize)
                }
            }
        }

        RandomAccessFile(part, "rw").use { outRaf ->
            outRaf.seek(HEADER_BYTES)
            val sb = ByteBuffer.allocate(rows * 4).order(ByteOrder.LITTLE_ENDIAN)
            sb.asFloatBuffer().put(scales)
            outRaf.write(sb.array())
        }

        check(part.length() == expectedSize) { "wrote ${part.length()} bytes, expected $expectedSize" }
        if (dst.exists()) dst.delete()
        if (!part.renameTo(dst)) error("cannot rename ${part.name} to ${dst.name}")
    }

    private fun q8(x: Float): Byte = Math.rint(x.toDouble()).coerceIn(-127.0, 127.0).toInt().toByte()
}

/** Read-only mmapped reader for `.q8` embedding tables produced by [EmbedTableQ8Converter]. */
class QuantizedEmbedTable private constructor(
    private val raf: RandomAccessFile,
    val rows: Int,
    val cols: Int,
    private val scales: FloatArray,
    private val weights: ByteBuffer,
) : Closeable {

    fun readRow(row: Int, out: FloatArray, outOffset: Int) {
        val s = scales[row]
        val base = row * cols
        for (c in 0 until cols) {
            out[outOffset + c] = weights.get(base + c).toInt() * s
        }
    }

    fun dotRow(row: Int, vec: FloatArray, vecOffset: Int): Float {
        val s = scales[row]
        val base = row * cols
        var sum = 0f
        for (c in 0 until cols) {
            sum += vec[vecOffset + c] * weights.get(base + c).toInt()
        }
        return sum * s
    }

    override fun close() {
        runCatching { raf.close() }
    }

    companion object {
        fun open(file: File): QuantizedEmbedTable {
            val raf = RandomAccessFile(file, "r")
            try {
                val hdr = ByteArray(EmbedTableQ8Converter.HEADER_BYTES.toInt())
                raf.readFully(hdr)
                val hb = ByteBuffer.wrap(hdr).order(ByteOrder.LITTLE_ENDIAN)
                val magic = hb.getInt()
                val version = hb.getInt()
                val rows = hb.getInt()
                val cols = hb.getInt()
                check(magic == EmbedTableQ8Converter.MAGIC && version == EmbedTableQ8Converter.VERSION) {
                    "Unsupported .q8 header in ${file.name}"
                }
                val weightsBytes = rows.toLong() * cols.toLong()
                check(weightsBytes <= Int.MAX_VALUE) { "Table too large for single mmap: $weightsBytes" }
                val scalesBytes = rows * 4
                val scalesBuf = ByteArray(scalesBytes)
                raf.readFully(scalesBuf)
                val scales = FloatArray(rows)
                ByteBuffer.wrap(scalesBuf).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(scales)
                val weightsOffset = EmbedTableQ8Converter.HEADER_BYTES + scalesBytes.toLong()
                val weights = raf.channel.map(FileChannel.MapMode.READ_ONLY, weightsOffset, weightsBytes)
                return QuantizedEmbedTable(raf, rows, cols, scales, weights)
            } catch (t: Throwable) {
                runCatching { raf.close() }
                throw t
            }
        }
    }
}
