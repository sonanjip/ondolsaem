package io.github.taehunkim.ondolsaem.engine

import io.github.taehunkim.ondolsaem.engine.iris.RunFiles
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

class TfliteW8ConverterTest {
    @get:Rule val tmp = TemporaryFolder()

    @Test fun fp16ToFloatHandlesNormalSubnormalZeroInfNan() {
        assertEquals(0f, TfliteW8Converter.fp16ToFloat(0x0000), 0f)
        assertEquals(-0f, TfliteW8Converter.fp16ToFloat(0x8000), 0f)
        assertEquals(1f, TfliteW8Converter.fp16ToFloat(0x3c00), 0f)
        assertEquals(-2f, TfliteW8Converter.fp16ToFloat(0xc000), 0f)
        assertEquals(65504f, TfliteW8Converter.fp16ToFloat(0x7bff), 0f)
        // Smallest normal: 2^-14 = 6.1035156e-5
        assertEquals(Math.pow(2.0, -14.0).toFloat(), TfliteW8Converter.fp16ToFloat(0x0400), 0f)
        // Largest subnormal: (1023/1024)*2^-14
        assertEquals((1023.0 / 1024.0 * Math.pow(2.0, -14.0)).toFloat(), TfliteW8Converter.fp16ToFloat(0x03ff), 1e-12f)
        // Smallest positive subnormal: 2^-24
        assertEquals(Math.pow(2.0, -24.0).toFloat(), TfliteW8Converter.fp16ToFloat(0x0001), 0f)
        assertEquals(Float.POSITIVE_INFINITY, TfliteW8Converter.fp16ToFloat(0x7c00), 0f)
        assertEquals(Float.NEGATIVE_INFINITY, TfliteW8Converter.fp16ToFloat(0xfc00), 0f)
        assertTrue(TfliteW8Converter.fp16ToFloat(0x7e00).isNaN())
    }

    @Test fun convertsInlineAndExternalDequantizeFcWeights() {
        // Test both small inline and large external (>= 256 KB) models.
        for (extraPayloadCount in listOf(4, 80_000)) {
            val outCh = 4
            val inCh = 6
            // Row 0: [1, -2, 0.5, 0, 0.25, -0.5], Row 1..3 varied
            val wFp16Shorts = shortArrayOf(
                0x3c00, 0xc000.toShort(), 0x3800, 0x0000, 0x3400, 0xb800.toShort(),
                0x4400, 0xbc00.toShort(), 0x0000, 0x3c00, 0xc200.toShort(), 0x3800,
                0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
                0xc400.toShort(), 0x4000, 0x3c00, 0xb800.toShort(), 0x3400, 0x0000,
            )
            val wBytes = ByteBuffer.allocate(wFp16Shorts.size * 2).order(ByteOrder.LITTLE_ENDIAN)
                .apply { for (s in wFp16Shorts) putShort(s) }.array()
            val extraBytes = ByteBuffer.allocate(extraPayloadCount * 4).order(ByteOrder.LITTLE_ENDIAN)
                .apply { for (i in 0 until extraPayloadCount) putFloat((i % 97) * 0.125f) }.array()
            val metaBytes = "2.3.0\u0000".toByteArray(Charsets.US_ASCII)

            val src = tmp.newFile("src_$extraPayloadCount.tflite")
            writeSyntheticTflite(src, outCh, inCh, wBytes, extraBytes, metaBytes)
            val dst = tmp.root.resolve("dst_$extraPayloadCount.tflite")

            val res = TfliteW8Converter.convert(src, dst)
            assertEquals(1, res.quantizedWeights)
            assertEquals(res.outputBytes, dst.length())

            // Parse converted FlatBuffer and verify structure & payloads.
            val dstBytes = dst.readBytes()
            val bb = ByteBuffer.wrap(dstBytes).order(ByteOrder.LITTLE_ENDIAN)
            assertEquals("TFL3", String(dstBytes, 4, 4, Charsets.US_ASCII))
            val root = bb.getInt(0)
            // Model -> subgraphs[0] (slot 8)
            val sgVecPos = indirect(bb, fieldPos(bb, root, 8))
            assertEquals(1, bb.getInt(sgVecPos))
            val sgPos = indirect(bb, sgVecPos + 4)

            // Operators: DEQUANTIZE removed, only ADD + FULLY_CONNECTED remain (2 ops).
            val opsVecPos = indirect(bb, fieldPos(bb, sgPos, 10))
            assertEquals(2, bb.getInt(opsVecPos))
            val fcPos = indirect(bb, opsVecPos + 4 + 4)
            val fcInsPos = indirect(bb, fieldPos(bb, fcPos, 6))
            // FULLY_CONNECTED input[1] was rewired from wDq (tensor 3) to wSrc (tensor 1).
            assertEquals(1, bb.getInt(fcInsPos + 4 + 4))

            // Verify tensor 1 is INT8 with 4 scales and 4 zeroPoints=0.
            val tensorsVecPos = indirect(bb, fieldPos(bb, sgPos, 4))
            val wTensorPos = indirect(bb, tensorsVecPos + 4 + 4)
            assertEquals(9, bb.get(fieldPos(bb, wTensorPos, 6)).toInt()) // TensorType.INT8 == 9
            val qPos = indirect(bb, fieldPos(bb, wTensorPos, 12))
            val scaleVecPos = indirect(bb, fieldPos(bb, qPos, 8))
            assertEquals(outCh, bb.getInt(scaleVecPos))
            val scale0 = bb.getFloat(scaleVecPos + 4)
            assertEquals(2f / 127f, scale0, 1e-7f)
            val scale2 = bb.getFloat(scaleVecPos + 4 + 8)
            assertEquals(1e-8f, scale2, 1e-12f) // zero row guard
            val zpVecPos = indirect(bb, fieldPos(bb, qPos, 10))
            assertEquals(outCh, bb.getInt(zpVecPos))
            for (r in 0 until outCh) assertEquals(0L, bb.getLong(zpVecPos + 4 + r * 8))

            // Verify Buffer 1 payload (inline for tiny, external 16-byte aligned for large).
            val bufsVecPos = indirect(bb, fieldPos(bb, root, 12))
            val buf1Pos = indirect(bb, bufsVecPos + 4 + 4)
            val buf2Pos = indirect(bb, bufsVecPos + 4 + 8)
            val wPayload = readBufBytes(bb, dstBytes, buf1Pos)
            assertEquals(outCh * inCh, wPayload.size)
            // Row 0: [1, -2, 0.5, 0, 0.25, -0.5] / (2/127) -> [64, -127, 32, 0, 16, -32]
            assertArrayEquals(byteArrayOf(64, -127, 32, 0, 16, -32), wPayload.copyOfRange(0, 6))
            // Unchanged buffer 2 preserved byte-for-byte.
            assertArrayEquals(extraBytes, readBufBytes(bb, dstBytes, buf2Pos))

            // Verify SignatureDef SubgraphIndex (slot 12) preserved.
            val sdVecPos = indirect(bb, fieldPos(bb, root, 18))
            val sdPos = indirect(bb, sdVecPos + 4)
            assertEquals(0, bb.getInt(fieldPos(bb, sdPos, 12)))
        }
    }

    @Test fun runFilesPrefersConvertedTfliteAndQ8Embed() {
        val dir = tmp.newFolder("iris")
        File(dir, "iris3b_config_dit_int8_fp16.json").writeText(
            """
            {
              "residency": "P1",
              "tokenizer": "iris_tokenizer.json",
              "embed": "iris_embed_bf16.f32",
              "host": { "file": "iris_host.f32", "tensors": {} },
              "graphs": {
                "te": [
                  { "role": "te00", "file": "iris_te_c00_f16w.tflite", "precision": "fp32", "signature": "serving_default", "inputs": ["a"], "outputs": [["o", 4]] }
                ],
                "adapter": { "role": "adapter", "file": "iris_dit_adapter_f16w_dit.tflite", "precision": "fp32", "signature": "serving_default", "inputs": ["a"], "outputs": [["o", 4]] },
                "dual": [
                  { "role": "dual0", "file": "iris_dit_dual0_i8w_dit.tflite", "precision": "default", "signature": "serving_default", "inputs": ["a"], "outputs": [["o", 4]] }
                ],
                "dual_mod_rows": [[0, 1]],
                "single": [],
                "single_mod_rows": [],
                "pixel": { "role": "pixel", "file": "iris_dit_pixel_f16w_dit.tflite", "precision": "default", "signature": "serving_default", "inputs": ["a"], "outputs": [["o", 4]] }
              }
            }
            """.trimIndent(),
        )
        File(dir, "iris_te_c00_i8w.tflite").writeBytes(ByteArray(123))
        File(dir, "iris_dit_adapter_i8w_dit.tflite").writeBytes(ByteArray(456))
        File(dir, "iris_dit_dual0_i8w_dit.tflite").writeBytes(ByteArray(789))
        File(dir, "iris_dit_pixel_f16w_dit.tflite").writeBytes(ByteArray(321))
        File(dir, "iris_embed_bf16.q8").writeBytes(ByteArray(64))

        val rf = RunFiles(dir)
        val cfg = rf.engineConfig()
        assertEquals("iris_te_c00_i8w.tflite", cfg.textEncoder[0].file)
        assertEquals(123L, cfg.textEncoder[0].bytes)
        assertEquals("iris_dit_adapter_i8w_dit.tflite", cfg.adapter.file)
        assertEquals(456L, cfg.adapter.bytes)
        assertEquals("iris_dit_dual0_i8w_dit.tflite", cfg.dual[0].file)
        assertEquals("iris_dit_pixel_f16w_dit.tflite", cfg.pixel.file)
        assertEquals("iris_embed_bf16.q8", rf.embedFile().name)
    }

    private fun readBufBytes(bb: ByteBuffer, fileBytes: ByteArray, bufPos: Int): ByteArray {
        val offPos = fieldPos(bb, bufPos, 6)
        if (offPos != 0) {
            val off = bb.getLong(offPos)
            val sz = bb.getLong(fieldPos(bb, bufPos, 8))
            assertTrue("external offset $off must be 16-byte aligned", off % 16L == 0L)
            return fileBytes.copyOfRange(off.toInt(), (off + sz).toInt())
        }
        val dPos = indirect(bb, fieldPos(bb, bufPos, 4))
        val len = bb.getInt(dPos)
        return fileBytes.copyOfRange(dPos + 4, dPos + 4 + len)
    }

    private fun indirect(bb: ByteBuffer, pos: Int): Int = pos + bb.getInt(pos)

    private fun fieldPos(bb: ByteBuffer, tablePos: Int, vtableByteOffset: Int): Int {
        val vtablePos = tablePos - bb.getInt(tablePos)
        val vtableSize = bb.getShort(vtablePos).toInt() and 0xffff
        if (vtableByteOffset >= vtableSize) return 0
        val rel = bb.getShort(vtablePos + vtableByteOffset).toInt() and 0xffff
        return if (rel == 0) 0 else tablePos + rel
    }

    /** Minimal FlatBuffer test generator for `[ADD(x, extra) -> DEQUANTIZE(w16) -> FULLY_CONNECTED]`. */
    private fun writeSyntheticTflite(
        dst: File,
        outCh: Int,
        inCh: Int,
        wFp16: ByteArray,
        extraF32: ByteArray,
        metaBytes: ByteArray,
    ) {
        val fb = TestFb()
        // Buffers: 0=empty, 1=wFp16, 2=extraF32, 3=metaBytes
        val b0 = fb.table(0) {}
        val b1 = fb.table(1) { addOff(0, fb.byteVec(wFp16)) }
        val b2 = fb.table(1) { addOff(0, fb.byteVec(extraF32)) }
        val b3 = fb.table(1) { addOff(0, fb.byteVec(metaBytes)) }
        val bufsVec = fb.offVec(intArrayOf(b0, b1, b2, b3))

        // OperatorCodes: 0=ADD(0), 1=DEQUANTIZE(6), 2=FULLY_CONNECTED(9)
        val oc0 = fb.table(4) { addByte(0, 0); addInt(2, 1); addInt(3, 0) }
        val oc1 = fb.table(4) { addByte(0, 6); addInt(2, 1); addInt(3, 6) }
        val oc2 = fb.table(4) { addByte(0, 9); addInt(2, 5); addInt(3, 9) }
        val ocsVec = fb.offVec(intArrayOf(oc0, oc1, oc2))

        // Tensors: 0=x [1,inCh], 1=w16 [outCh,inCh] (buf 1), 2=extra [extraN] (buf 2), 3=w_dq [outCh,inCh], 4=add_out [1,inCh], 5=y [1,outCh]
        fun tensor(name: String, shape: IntArray, type: Byte, buf: Int): Int {
            val sOff = fb.intVec(shape)
            val nOff = fb.str(name)
            return fb.table(9) {
                addOff(0, sOff)
                addByte(1, type)
                if (buf != 0) addInt(2, buf)
                addOff(3, nOff)
                addByte(8, 1)
            }
        }
        val t0 = tensor("x", intArrayOf(1, inCh), 0, 0)
        val t1 = tensor("w16", intArrayOf(outCh, inCh), 1, 1)
        val t2 = tensor("extra", intArrayOf(extraF32.size / 4), 0, 2)
        val t3 = tensor("w16_dequant", intArrayOf(outCh, inCh), 0, 0)
        val t4 = tensor("h", intArrayOf(1, inCh), 0, 0)
        val t5 = tensor("y", intArrayOf(1, outCh), 0, 0)
        val tensorsVec = fb.offVec(intArrayOf(t0, t1, t2, t3, t4, t5))

        val addOpts = fb.table(2) { addByte(1, 1) } // potScaleInt16 = true
        val fcOpts = fb.table(4) { addByte(2, 1) }  // keepNumDims = true
        val opAdd = fb.table(5) {
            addInt(0, 0)
            addOff(1, fb.intVec(intArrayOf(0, 2)))
            addOff(2, fb.intVec(intArrayOf(4)))
            addByte(3, 11) // BuiltinOptions.AddOptions
            addOff(4, addOpts)
        }
        val opDq = fb.table(3) {
            addInt(0, 1)
            addOff(1, fb.intVec(intArrayOf(1)))
            addOff(2, fb.intVec(intArrayOf(3)))
        }
        val opFc = fb.table(5) {
            addInt(0, 2)
            addOff(1, fb.intVec(intArrayOf(4, 3)))
            addOff(2, fb.intVec(intArrayOf(5)))
            addByte(3, 8) // FullyConnectedOptions
            addOff(4, fcOpts)
        }
        val opsVec = fb.offVec(intArrayOf(opAdd, opDq, opFc))

        val sg = fb.table(4) {
            addOff(0, tensorsVec)
            addOff(1, fb.intVec(intArrayOf(0)))
            addOff(2, fb.intVec(intArrayOf(5)))
            addOff(3, opsVec)
        }
        val sgsVec = fb.offVec(intArrayOf(sg))

        val md = fb.table(2) { addOff(0, fb.str("min_runtime_version")); addInt(1, 3) }
        val mdVec = fb.offVec(intArrayOf(md))

        val sd = fb.table(5) {
            addOff(0, fb.offVec(intArrayOf(fb.table(2) { addOff(0, fb.str("x")) })))
            addOff(1, fb.offVec(intArrayOf(fb.table(2) { addOff(0, fb.str("y")); addInt(1, 5) })))
            addOff(2, fb.str("serving_default"))
            addInt(4, 0)
        }
        val sdVec = fb.offVec(intArrayOf(sd))

        val model = fb.table(8) {
            addInt(0, 3)
            addOff(1, ocsVec)
            addOff(2, sgsVec)
            addOff(4, bufsVec)
            addOff(6, mdVec)
            addOff(7, sdVec)
        }
        dst.writeBytes(fb.finish(model))
    }

    private class TestFb {
        private var buf = ByteArray(1 shl 20)
        private var space = buf.size
        private var slots = IntArray(0)
        private var bottom = 0
        fun off() = buf.size - space
        private fun ensure(n: Int) {
            if (space >= n) return
            var cap = buf.size; val used = off(); while (cap - used < n) cap = cap shl 1
            val next = ByteArray(cap); System.arraycopy(buf, space, next, cap - used, used); buf = next; space = cap - used
        }
        private fun prep(align: Int, add: Int) {
            val m = (off() + add) % align; val p = if (m == 0) 0 else align - m; ensure(p); repeat(p) { buf[--space] = 0 }
        }
        fun putByte(v: Byte) { ensure(1); buf[--space] = v }
        fun putShort(v: Short) { prep(2, 0); ensure(2); buf[--space] = (v.toInt() ushr 8).toByte(); buf[--space] = v.toByte() }
        fun putInt(v: Int) { prep(4, 0); ensure(4); for (k in 3 downTo 0) buf[--space] = (v ushr (k * 8)).toByte() }
        fun putBytes(b: ByteArray) { ensure(b.size); space -= b.size; System.arraycopy(b, 0, buf, space, b.size) }
        fun putOff(t: Int) { prep(4, 0); putInt((off() + 4) - t) }
        fun byteVec(b: ByteArray): Int { prep(16, b.size); putBytes(b); putInt(b.size); return off() }
        fun str(s: String): Int { val b = s.toByteArray(); prep(4, b.size + 1); putByte(0); putBytes(b); putInt(b.size); return off() }
        fun intVec(a: IntArray): Int { prep(4, a.size * 4); for (i in a.size - 1 downTo 0) putInt(a[i]); putInt(a.size); return off() }
        fun offVec(a: IntArray): Int { prep(4, a.size * 4); for (i in a.size - 1 downTo 0) putOff(a[i]); putInt(a.size); return off() }
        fun addByte(i: Int, v: Byte) { putByte(v); slots[i] = off() }
        fun addInt(i: Int, v: Int) { putInt(v); slots[i] = off() }
        fun addOff(i: Int, t: Int) { putOff(t); slots[i] = off() }
        fun table(n: Int, block: TestFb.() -> Unit): Int {
            val prevSlots = slots; val prevBottom = bottom
            slots = IntArray(n); bottom = off(); block(); putInt(0); val ts = off(); val os = ts - bottom
            for (i in n - 1 downTo 0) putShort((if (slots[i] == 0) 0 else ts - slots[i]).toShort())
            putShort(os.toShort()); putShort((4 + n * 2).toShort())
            val vo = off(); val sp = buf.size - ts; val soff = vo - ts
            for (k in 0..3) buf[sp + k] = (soff ushr (k * 8)).toByte()
            slots = prevSlots; bottom = prevBottom
            return ts
        }
        fun finish(root: Int): ByteArray { prep(16, 8); putBytes("TFL3".toByteArray()); putOff(root); return buf.copyOfRange(space, buf.size) }
    }
}
