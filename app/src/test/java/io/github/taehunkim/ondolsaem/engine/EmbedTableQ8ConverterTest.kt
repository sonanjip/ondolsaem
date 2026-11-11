package io.github.taehunkim.ondolsaem.engine

import io.github.taehunkim.ondolsaem.engine.iris.PromptTokens
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.abs

class EmbedTableQ8ConverterTest {
    @get:Rule val tmp = TemporaryFolder()

    private fun floatToBf16Short(f: Float): Short = (java.lang.Float.floatToRawIntBits(f) ushr 16).toShort()

    private fun writeNpyBf16(file: File, rows: Int, cols: Int, values: FloatArray) {
        val rawDict = "{'descr': '<u2', 'fortran_order': False, 'shape': ($rows, $cols), }"
        val pad = (16 - (10 + rawDict.length + 1) % 16) % 16
        val dict = rawDict + " ".repeat(pad) + "\n"
        val hdr = ByteArray(10 + dict.length)
        hdr[0] = 0x93.toByte()
        "NUMPY".toByteArray(Charsets.US_ASCII).copyInto(hdr, 1)
        hdr[6] = 1
        hdr[7] = 0
        hdr[8] = (dict.length and 0xff).toByte()
        hdr[9] = ((dict.length ushr 8) and 0xff).toByte()
        dict.toByteArray(Charsets.US_ASCII).copyInto(hdr, 10)

        val data = ByteBuffer.allocate(values.size * 2).order(ByteOrder.LITTLE_ENDIAN)
        for (v in values) data.putShort(floatToBf16Short(v))
        file.writeBytes(hdr + data.array())
    }

    @Test fun convertsNpyAndReadsRowsAndDotProduct() {
        val rows = 3
        val cols = 4
        val values = floatArrayOf(
            1f, -2f, 0.5f, 0f,
            0f, 0f, 0f, 0f,
            -3f, 1.5f, 0.25f, -0.75f,
        )
        val src = tmp.newFile("embed_table.npy")
        writeNpyBf16(src, rows, cols, values)
        val dst = tmp.root.resolve("embed_table.q8")

        assertFalse(EmbedTableQ8Converter.isQ8File(src))
        EmbedTableQ8Converter.convertNpy(src, dst)
        assertTrue(EmbedTableQ8Converter.isQ8File(dst))
        assertEquals(16L + rows * 4L + rows * cols, dst.length())

        QuantizedEmbedTable.open(dst).use { qt ->
            assertEquals(rows, qt.rows)
            assertEquals(cols, qt.cols)

            val row0 = FloatArray(cols)
            qt.readRow(0, row0, 0)
            for (c in 0 until cols) assertEquals(values[c], row0[c], 0.02f)

            val row1 = FloatArray(cols)
            qt.readRow(1, row1, 0)
            assertArrayEquals(floatArrayOf(0f, 0f, 0f, 0f), row1, 0f)

            val vec = floatArrayOf(0f, 2f, -1f, 0.5f, 4f)
            val expectedDot = (-3f * 2f) + (1.5f * -1f) + (0.25f * 0.5f) + (-0.75f * 4f)
            assertEquals(expectedDot, qt.dotRow(2, vec, 1), 0.08f)
        }
    }

    @Test fun convertsRawBf16TableAcrossMultipleRows() {
        val rows = 5
        val cols = 8
        val values = FloatArray(rows * cols) { i -> ((i % 13) - 6) * 0.25f }
        val src = tmp.newFile("iris_embed_bf16.f32")
        val bb = ByteBuffer.allocate(values.size * 2).order(ByteOrder.LITTLE_ENDIAN)
        for (v in values) bb.putShort(floatToBf16Short(v))
        src.writeBytes(bb.array())

        val dst = tmp.root.resolve("iris_embed_bf16.q8")
        EmbedTableQ8Converter.convertRawBf16(src, cols, dst)
        assertTrue(EmbedTableQ8Converter.isQ8File(dst))

        QuantizedEmbedTable.open(dst).use { qt ->
            val buf = FloatArray(cols)
            for (r in 0 until rows) {
                qt.readRow(r, buf, 0)
                for (c in 0 until cols) {
                    val expected = Float.fromBits((floatToBf16Short(values[r * cols + c]).toInt() and 0xffff) shl 16)
                    assertTrue(abs(expected - buf[c]) <= 0.02f)
                }
            }
        }
    }

    @Test fun promptTokensReadsQ8AndFallsBackToBf16() {
        val assets = tmp.newFolder("tok")
        File(assets, "qwen_vocab.txt").writeText("h\ne\nl\no\n")
        File(assets, "qwen_merges.txt").writeText("")
        File(assets, "prompt_prefix_ids.txt").writeText("0, 1")
        File(assets, "prompt_suffix_ids.txt").writeText("2, 3")

        val rows = 4
        val cols = PromptTokens.HIDDEN
        val bb = ByteBuffer.allocate(rows * cols * 2).order(ByteOrder.LITTLE_ENDIAN)
        for (r in 0 until rows) {
            for (c in 0 until cols) {
                val v = ((r + 1) * 0.25f) * (if (c % 2 == 0) 1f else -0.5f)
                bb.putShort(floatToBf16Short(v))
            }
        }
        val rawFile = tmp.newFile("iris_embed_bf16.f32").apply { writeBytes(bb.array()) }
        val q8File = tmp.root.resolve("iris_embed_bf16.q8")
        EmbedTableQ8Converter.convertRawBf16(rawFile, cols, q8File)

        val tokens = PromptTokens.Tokens(IntArray(PromptTokens.SEQ_LEN) { it % rows }, 4, 4)
        val bf16Out = PromptTokens(assets, rawFile).use { it.embed(tokens) }
        val q8Out = PromptTokens(assets, q8File).use { it.embed(tokens) }
        assertEquals(bf16Out.size, q8Out.size)
        for (i in bf16Out.indices) assertEquals(bf16Out[i], q8Out[i], 0.02f)
    }
}
