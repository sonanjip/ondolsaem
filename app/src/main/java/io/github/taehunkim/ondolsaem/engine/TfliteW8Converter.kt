package io.github.taehunkim.ondolsaem.engine

import java.io.BufferedOutputStream
import java.io.File
import java.io.FileOutputStream
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.abs

/**
 * Converts FP16 weight-only `.tflite` models (where `FULLY_CONNECTED` weights are stored as
 * `Const(FLOAT16) -> DEQUANTIZE -> FULLY_CONNECTED`) into channel-wise INT8 weights wired directly
 * into `FULLY_CONNECTED` (bypassing and removing the `DEQUANTIZE` op, matching `ai-edge-quantizer`'s
 * dynamic INT8 graph format that runs on both LiteRT GPU and XNNPACK CPU).
 *
 * Zero APK assets required: parses the source `.tflite` FlatBuffer directly, supports >2 GiB models
 * using 64-bit external buffer offsets (`Buffer.offset` / `Buffer.size`), and streams payloads in
 * bounded memory.
 */
object TfliteW8Converter {
    private const val TENSOR_FLOAT32 = 0
    private const val TENSOR_FLOAT16 = 1
    private const val TENSOR_INT8 = 9

    private const val OP_DEQUANTIZE = 6
    private const val OP_FULLY_CONNECTED = 9

    private const val BO_NONE = 0
    private const val BO_CONCAT_EMBEDDINGS = 3
    private const val BO_FULLY_CONNECTED = 8
    private const val BO_RESHAPE = 17
    private const val BO_SQUEEZE = 30
    private const val BO_VAR_HANDLE = 111
    private const val BO_BUCKETIZE = 115

    private const val INLINE_LIMIT = 1024L
    private const val EXTERNAL_THRESHOLD = 256L * 1024L
    private const val CHUNK = 4 shl 20

    private class OpCodeSpec(
        val deprecatedBuiltinCode: Byte,
        val customCode: String?,
        val version: Int,
        val builtinCode: Int,
        val hasVersion: Boolean,
        val hasBuiltinCode: Boolean,
    ) {
        val effectiveCode: Int get() = maxOf(builtinCode, deprecatedBuiltinCode.toInt())
    }

    private class QuantSpec(
        val min: FloatArray?,
        val max: FloatArray?,
        val scale: FloatArray?,
        val zeroPoint: LongArray?,
        val detailsType: Int,
        val detailsTablePos: Long,
        val quantizedDimension: Int,
        val hasQuantizedDimension: Boolean,
    )

    private class TensorSpec(
        val shape: IntArray?,
        var type: Byte,
        val hasType: Boolean,
        var buffer: Int,
        val hasBuffer: Boolean,
        val name: String?,
        var quantization: QuantSpec?,
        val isVariable: Boolean,
        val hasIsVariable: Boolean,
        val shapeSignature: IntArray?,
        val hasRank: Boolean,
        val hasHasRank: Boolean,
        val externalBuffer: Int,
        val hasExternalBuffer: Boolean,
    )

    private class OpSpec(
        val opcodeIndex: Int,
        val inputs: IntArray?,
        val outputs: IntArray?,
        val builtinOptionsType: Int,
        val builtinOptionsPos: Long,
        val customOptions: ByteArray?,
        val customOptionsFormat: Byte,
        val hasCustomOptionsFormat: Boolean,
        val mutatingVariableInputs: ByteArray?,
        val intermediates: IntArray?,
        val largeCustomOptionsOffset: Long,
        val largeCustomOptionsSize: Long,
        val builtinOptions2Type: Int,
        val builtinOptions2Pos: Long,
        val debugMetadataIndex: Int,
        val hasDebugMetadataIndex: Boolean,
    )

    private class SubGraphSpec(
        val tensors: MutableList<TensorSpec>,
        val inputs: IntArray?,
        val outputs: IntArray?,
        val operators: MutableList<OpSpec>,
        val name: String?,
        val debugMetadataIndex: Int,
        val hasDebugMetadataIndex: Boolean,
    )

    private class BufferSpec(
        val dataPos: Long,
        val dataLen: Int,
        val extOffset: Long,
        val extSize: Long,
    ) {
        val isExternal: Boolean get() = extOffset > 1L && extSize > 0L
        val payloadSize: Long get() = if (isExternal) extSize else dataLen.toLong()
        val payloadOffset: Long get() = if (isExternal) extOffset else dataPos
    }

    private class MetadataSpec(val name: String?, val buffer: Int)
    private class TensorMapSpec(val name: String?, val tensorIndex: Int)
    private class SignatureDefSpec(
        val inputs: List<TensorMapSpec>,
        val outputs: List<TensorMapSpec>,
        val signatureKey: String?,
        val subgraphIndex: Int,
        val hasSubgraphIndex: Boolean,
    )

    private class ModelSpec(
        val version: Int,
        val operatorCodes: List<OpCodeSpec>,
        val subgraphs: List<SubGraphSpec>,
        val description: String?,
        val buffers: List<BufferSpec>,
        val metadataBuffer: IntArray?,
        val metadata: List<MetadataSpec>,
        val signatureDefs: List<SignatureDefSpec>,
    )

    private class QuantTarget(
        val bufferIdx: Int,
        val outCh: Int,
        val inCh: Int,
        var scales: FloatArray = FloatArray(0),
    )

    data class Result(val quantizedWeights: Int, val outputBytes: Long)

    fun convert(src: File, dst: File, onProgress: (Long, Long) -> Unit = { _, _ -> }): Result {
        dst.parentFile?.mkdirs()
        val part = File(dst.path + ".part")
        var quantizedCount = 0
        var finalBytes = 0L

        RandomAccessFile(src, "r").use { raf ->
            val reader = FbReader(raf)
            val model = readModel(reader)
            val quantTargets = LinkedHashMap<Int, QuantTarget>() // bufferIdx -> target

            val metadataBuffers = HashSet<Int>()
            model.metadata.forEach { metadataBuffers.add(it.buffer) }

            // Count how many tensors point to each non-zero buffer so we never mutate shared buffers.
            val bufferRefCount = HashMap<Int, Int>()
            for (sg in model.subgraphs) {
                for (t in sg.tensors) {
                    if (t.buffer > 0) bufferRefCount[t.buffer] = (bufferRefCount[t.buffer] ?: 0) + 1
                }
            }

            for (sg in model.subgraphs) {
                val consumers = HashMap<Int, MutableList<Pair<Int, Int>>>() // tensorIdx -> (opIdx, inputSlot)
                sg.operators.forEachIndexed { oi, op ->
                    op.inputs?.forEachIndexed { slot, ti ->
                        if (ti >= 0) consumers.getOrPut(ti) { ArrayList() }.add(oi to slot)
                    }
                }
                val graphIo = HashSet<Int>()
                sg.inputs?.forEach { graphIo.add(it) }
                sg.outputs?.forEach { graphIo.add(it) }

                val opsToRemove = HashSet<Int>()
                for ((oi, op) in sg.operators.withIndex()) {
                    if (model.operatorCodes[op.opcodeIndex].effectiveCode != OP_DEQUANTIZE) continue
                    val ins = op.inputs ?: continue
                    val outs = op.outputs ?: continue
                    if (ins.size != 1 || outs.size != 1) continue
                    val wSrcIdx = ins[0]
                    val wDqIdx = outs[0]
                    if (wSrcIdx < 0 || wDqIdx < 0) continue
                    if (wSrcIdx in graphIo || wDqIdx in graphIo) continue

                    val wSrc = sg.tensors[wSrcIdx]
                    if (wSrc.type.toInt() != TENSOR_FLOAT16) continue
                    val shape = wSrc.shape ?: continue
                    if (shape.size != 2 || shape[0] <= 0 || shape[1] <= 0) continue
                    if (wSrc.buffer <= 0 || wSrc.buffer in metadataBuffers || (bufferRefCount[wSrc.buffer] ?: 0) != 1) continue
                    val bufSpec = model.buffers[wSrc.buffer]
                    val expectedFp16Bytes = shape[0].toLong() * shape[1].toLong() * 2L
                    if (bufSpec.payloadSize != expectedFp16Bytes) continue

                    val srcCons = consumers[wSrcIdx] ?: continue
                    if (srcCons.size != 1) continue
                    val dqCons = consumers[wDqIdx] ?: continue
                    if (dqCons.isEmpty()) continue
                    val allFcWeight = dqCons.all { (cOi, slot) ->
                        slot == 1 && model.operatorCodes[sg.operators[cOi].opcodeIndex].effectiveCode == OP_FULLY_CONNECTED
                    }
                    if (!allFcWeight) continue

                    // Rewire every consumer FULLY_CONNECTED op directly to the INT8 tensor and remove DEQUANTIZE.
                    for ((cOi, slot) in dqCons) {
                        sg.operators[cOi].inputs!![slot] = wSrcIdx
                    }
                    opsToRemove.add(oi)
                    sg.tensors[wDqIdx].buffer = 0
                    quantTargets[wSrc.buffer] = QuantTarget(wSrc.buffer, shape[0], shape[1])
                }

                if (opsToRemove.isNotEmpty()) {
                    val kept = ArrayList<OpSpec>(sg.operators.size - opsToRemove.size)
                    sg.operators.forEachIndexed { oi, op -> if (oi !in opsToRemove) kept.add(op) }
                    sg.operators.clear()
                    sg.operators.addAll(kept)
                }
            }

            quantizedCount = quantTargets.size

            // Pass 1: Compute per-row scales for all quantized FP16 weights in bounded memory.
            val totalFp16Bytes = quantTargets.values.sumOf { it.outCh.toLong() * it.inCh.toLong() * 2L }
            var pass1Done = 0L
            for ((bufIdx, qt) in quantTargets) {
                qt.scales = computeRowScales(raf, model.buffers[bufIdx], qt.outCh, qt.inCh) { bytesRead ->
                    onProgress((pass1Done + bytesRead) / 3, maxOf(1L, totalFp16Bytes))
                }
                pass1Done += qt.outCh.toLong() * qt.inCh.toLong() * 2L
            }

            // Update tensor metadata with INT8 type + per-channel QuantizationParameters.
            for (sg in model.subgraphs) {
                for (t in sg.tensors) {
                    val qt = quantTargets[t.buffer] ?: continue
                    t.type = TENSOR_INT8.toByte()
                    t.quantization = QuantSpec(
                        min = null,
                        max = null,
                        scale = qt.scales,
                        zeroPoint = LongArray(qt.outCh),
                        detailsType = 0,
                        detailsTablePos = 0L,
                        quantizedDimension = 0,
                        hasQuantizedDimension = true,
                    )
                }
            }

            // Decide whether large buffers should be packed outside the FlatBuffer (16-byte aligned).
            val hasExternal = model.buffers.any { it.isExternal }
            val totalPayload = model.buffers.indices.sumOf { i ->
                val qt = quantTargets[i]
                qt?.let { it.outCh.toLong() * it.inCh.toLong() } ?: model.buffers[i].payloadSize
            }
            val useExternal = hasExternal || totalPayload >= EXTERNAL_THRESHOLD

            val isExternalBuf = BooleanArray(model.buffers.size) { i ->
                if (!useExternal || i == 0 || i in metadataBuffers) false
                else {
                    val size = quantTargets[i]?.let { it.outCh.toLong() * it.inCh.toLong() } ?: model.buffers[i].payloadSize
                    model.buffers[i].isExternal || size >= INLINE_LIMIT
                }
            }

            val inlineData = arrayOfNulls<ByteArray>(model.buffers.size)
            for (i in model.buffers.indices) {
                if (i == 0 || isExternalBuf[i]) continue
                val qt = quantTargets[i]
                if (qt != null) {
                    inlineData[i] = quantizeAllToBytes(raf, model.buffers[i], qt)
                } else {
                    val b = model.buffers[i]
                    if (b.payloadSize > 0) inlineData[i] = reader.bytes(b.payloadOffset, b.payloadSize.toInt())
                }
            }

            val bufTableOffsets = IntArray(model.buffers.size)
            val headerBytes = writeModelHeader(reader, model, isExternalBuf, inlineData, bufTableOffsets)
            val headerPaddedLen = roundUp16(headerBytes.size.toLong())

            // Compute exact 16-byte-aligned external buffer offsets and sizes, then patch headerBytes in RAM.
            var cursor = headerPaddedLen
            val extOffsets = LongArray(model.buffers.size)
            val extSizes = LongArray(model.buffers.size)
            val hb = ByteBuffer.wrap(headerBytes).order(ByteOrder.LITTLE_ENDIAN)
            for (i in model.buffers.indices) {
                if (!isExternalBuf[i]) continue
                val sz = quantTargets[i]?.let { it.outCh.toLong() * it.inCh.toLong() } ?: model.buffers[i].payloadSize
                extOffsets[i] = cursor
                extSizes[i] = sz
                patchBufferTable(hb, headerBytes.size - bufTableOffsets[i], cursor, sz)
                cursor = roundUp16(cursor + sz)
            }
            finalBytes = cursor

            // Pass 2: Stream header + external buffers sequentially to dst.part.
            val totalWriteBytes = finalBytes
            var writtenBytes = 0L
            BufferedOutputStream(FileOutputStream(part), 1 shl 20).use { out ->
                out.write(headerBytes)
                val hdrPad = (headerPaddedLen - headerBytes.size).toInt()
                if (hdrPad > 0) out.write(ByteArray(hdrPad))
                writtenBytes = headerPaddedLen

                val zeroPad = ByteArray(16)
                for (i in model.buffers.indices) {
                    if (!isExternalBuf[i]) continue
                    val qt = quantTargets[i]
                    if (qt != null) {
                        streamQuantizeFp16ToInt8(raf, model.buffers[i], qt, out) { done ->
                            val w = writtenBytes + done
                            onProgress(totalWriteBytes / 3 + (w * 2) / 3, totalWriteBytes)
                        }
                    } else {
                        copyRegion(raf, model.buffers[i].payloadOffset, model.buffers[i].payloadSize, out)
                    }
                    val sz = extSizes[i]
                    val pad = (roundUp16(sz) - sz).toInt()
                    if (pad > 0) out.write(zeroPad, 0, pad)
                    writtenBytes += sz + pad
                    onProgress(totalWriteBytes / 3 + (writtenBytes * 2) / 3, totalWriteBytes)
                }
            }
        }

        check(part.length() == finalBytes) { "wrote ${part.length()} bytes, expected $finalBytes" }
        if (dst.exists()) dst.delete()
        if (!part.renameTo(dst)) error("cannot rename ${part.name} to ${dst.name}")
        onProgress(finalBytes, finalBytes)
        return Result(quantizedCount, finalBytes)
    }

    private fun computeRowScales(
        raf: RandomAccessFile,
        buf: BufferSpec,
        outCh: Int,
        inCh: Int,
        onBytes: (Long) -> Unit,
    ): FloatArray {
        val scales = FloatArray(outCh)
        val rowsPerChunk = maxOf(1, CHUNK / (inCh * 2))
        val bb = ByteBuffer.allocate(rowsPerChunk * inCh * 2).order(ByteOrder.LITTLE_ENDIAN)
        val ch = raf.channel
        var r0 = 0
        while (r0 < outCh) {
            val count = minOf(rowsPerChunk, outCh - r0)
            val byteLen = count * inCh * 2
            bb.clear()
            bb.limit(byteLen)
            var pos = buf.payloadOffset + r0.toLong() * inCh * 2L
            while (bb.hasRemaining()) {
                val n = ch.read(bb, pos)
                check(n > 0) { "Unexpected EOF computing scales" }
                pos += n
            }
            bb.flip()
            val sb = bb.asShortBuffer()
            for (i in 0 until count) {
                var maxAbs = 0f
                for (j in 0 until inCh) {
                    val v = fp16ToFloat(sb.get().toInt() and 0xffff)
                    val a = abs(v)
                    if (a > maxAbs) maxAbs = a
                }
                scales[r0 + i] = if (maxAbs == 0f) 1e-8f else maxAbs / 127f
            }
            r0 += count
            onBytes(r0.toLong() * inCh * 2L)
        }
        return scales
    }

    private fun quantizeAllToBytes(raf: RandomAccessFile, buf: BufferSpec, qt: QuantTarget): ByteArray {
        val out = ByteArray(qt.outCh * qt.inCh)
        val raw = ByteArray(qt.outCh * qt.inCh * 2)
        raf.seek(buf.payloadOffset)
        raf.readFully(raw)
        val sb = ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN).asShortBuffer()
        for (r in 0 until qt.outCh) {
            val sc = qt.scales[r]
            val base = r * qt.inCh
            for (c in 0 until qt.inCh) {
                out[base + c] = q8(fp16ToFloat(sb.get().toInt() and 0xffff) / sc)
            }
        }
        return out
    }

    private fun streamQuantizeFp16ToInt8(
        raf: RandomAccessFile,
        buf: BufferSpec,
        qt: QuantTarget,
        out: BufferedOutputStream,
        onBytes: (Long) -> Unit,
    ) {
        val rowsPerChunk = maxOf(1, CHUNK / (qt.inCh * 2))
        val bb = ByteBuffer.allocate(rowsPerChunk * qt.inCh * 2).order(ByteOrder.LITTLE_ENDIAN)
        val outBytes = ByteArray(rowsPerChunk * qt.inCh)
        val ch = raf.channel
        var r0 = 0
        while (r0 < qt.outCh) {
            val count = minOf(rowsPerChunk, qt.outCh - r0)
            val byteLen = count * qt.inCh * 2
            bb.clear()
            bb.limit(byteLen)
            var pos = buf.payloadOffset + r0.toLong() * qt.inCh * 2L
            while (bb.hasRemaining()) {
                val n = ch.read(bb, pos)
                check(n > 0) { "Unexpected EOF quantizing buffer" }
                pos += n
            }
            bb.flip()
            val sb = bb.asShortBuffer()
            for (i in 0 until count) {
                val sc = qt.scales[r0 + i]
                val base = i * qt.inCh
                for (c in 0 until qt.inCh) {
                    outBytes[base + c] = q8(fp16ToFloat(sb.get().toInt() and 0xffff) / sc)
                }
            }
            out.write(outBytes, 0, count * qt.inCh)
            r0 += count
            onBytes(r0.toLong() * qt.inCh)
        }
    }

    private fun copyRegion(raf: RandomAccessFile, offset: Long, length: Long, out: BufferedOutputStream) {
        val buf = ByteArray(CHUNK)
        var done = 0L
        val ch = raf.channel
        val bb = ByteBuffer.wrap(buf)
        while (done < length) {
            val want = minOf(CHUNK.toLong(), length - done).toInt()
            bb.clear()
            bb.limit(want)
            var pos = offset + done
            while (bb.hasRemaining()) {
                val n = ch.read(bb, pos)
                check(n > 0) { "Unexpected EOF copying buffer" }
                pos += n
            }
            out.write(buf, 0, want)
            done += want
        }
    }

    internal fun fp16ToFloat(h: Int): Float {
        val s = (h and 0x8000) shl 16
        val exp = (h ushr 10) and 0x1f
        val mant = h and 0x03ff
        val bits = when (exp) {
            0 -> if (mant == 0) s else {
                var m = mant
                var e = -14
                while ((m and 0x0400) == 0) {
                    m = m shl 1
                    e--
                }
                m = m and 0x03ff
                s or ((e + 127) shl 23) or (m shl 13)
            }
            0x1f -> s or 0x7f800000 or (mant shl 13)
            else -> s or ((exp - 15 + 127) shl 23) or (mant shl 13)
        }
        return Float.fromBits(bits)
    }

    private fun q8(x: Float): Byte = Math.rint(x.toDouble()).coerceIn(-127.0, 127.0).toInt().toByte()

    private fun roundUp16(v: Long): Long = (v + 15L) and 15L.inv()

    private fun patchBufferTable(bb: ByteBuffer, tablePos: Int, offset: Long, size: Long) {
        val vtablePos = tablePos - bb.getInt(tablePos)
        val vtableSize = bb.getShort(vtablePos).toInt() and 0xffff
        check(vtableSize >= 10) { "Buffer vtable too small: $vtableSize" }
        val offSlot = bb.getShort(vtablePos + 6).toInt() and 0xffff
        val szSlot = bb.getShort(vtablePos + 8).toInt() and 0xffff
        check(offSlot > 0 && szSlot > 0) { "Missing offset/size slots in Buffer" }
        bb.putLong(tablePos + offSlot, offset)
        bb.putLong(tablePos + szSlot, size)
    }

    private fun readModel(r: FbReader): ModelSpec {
        val modelPos = r.u32(0L)
        check(r.bytes(4L, 4).contentEquals("TFL3".toByteArray())) { "Not a TFLite FlatBuffer (missing TFL3)" }
        check(r.fieldPos(modelPos, 20) == 0L && r.fieldPos(modelPos, 22) == 0L) {
            "Unsupported ExternalBufferGroups/ExternalBuffers in TFLite model"
        }

        val version = r.fieldPos(modelPos, 4).let { if (it != 0L) r.i32(it) else 0 }

        val opCodes = ArrayList<OpCodeSpec>()
        val ocVec = r.fieldPos(modelPos, 6)
        if (ocVec != 0L) {
            val n = r.vecLen(ocVec)
            for (i in 0 until n) opCodes.add(readOpCode(r, r.vecTable(ocVec, i)))
        }

        val subgraphs = ArrayList<SubGraphSpec>()
        val sgVec = r.fieldPos(modelPos, 8)
        if (sgVec != 0L) {
            val n = r.vecLen(sgVec)
            for (i in 0 until n) subgraphs.add(readSubGraph(r, r.vecTable(sgVec, i)))
        }

        val description = r.fieldPos(modelPos, 10).let { if (it != 0L) r.str(it) else null }

        val buffers = ArrayList<BufferSpec>()
        val bufVec = r.fieldPos(modelPos, 12)
        if (bufVec != 0L) {
            val n = r.vecLen(bufVec)
            for (i in 0 until n) buffers.add(readBuffer(r, r.vecTable(bufVec, i)))
        }

        val metadataBuffer = r.fieldPos(modelPos, 14).let { if (it != 0L) r.intVec(it) else null }

        val metadata = ArrayList<MetadataSpec>()
        val mdVec = r.fieldPos(modelPos, 16)
        if (mdVec != 0L) {
            val n = r.vecLen(mdVec)
            for (i in 0 until n) {
                val p = r.vecTable(mdVec, i)
                val name = r.fieldPos(p, 4).let { if (it != 0L) r.str(it) else null }
                val buf = r.fieldPos(p, 6).let { if (it != 0L) r.i32(it) else 0 }
                metadata.add(MetadataSpec(name, buf))
            }
        }

        val sigDefs = ArrayList<SignatureDefSpec>()
        val sdVec = r.fieldPos(modelPos, 18)
        if (sdVec != 0L) {
            val n = r.vecLen(sdVec)
            for (i in 0 until n) sigDefs.add(readSignatureDef(r, r.vecTable(sdVec, i)))
        }

        return ModelSpec(version, opCodes, subgraphs, description, buffers, metadataBuffer, metadata, sigDefs)
    }

    private fun readOpCode(r: FbReader, pos: Long): OpCodeSpec {
        val dep = r.fieldPos(pos, 4).let { if (it != 0L) r.i8(it) else 0 }
        val custom = r.fieldPos(pos, 6).let { if (it != 0L) r.str(it) else null }
        val verPos = r.fieldPos(pos, 8)
        val ver = if (verPos != 0L) r.i32(verPos) else 1
        val bcPos = r.fieldPos(pos, 10)
        val bc = if (bcPos != 0L) r.i32(bcPos) else 0
        return OpCodeSpec(dep, custom, ver, bc, verPos != 0L, bcPos != 0L)
    }

    private fun readSubGraph(r: FbReader, pos: Long): SubGraphSpec {
        val tensors = ArrayList<TensorSpec>()
        val tVec = r.fieldPos(pos, 4)
        if (tVec != 0L) {
            val n = r.vecLen(tVec)
            for (i in 0 until n) tensors.add(readTensor(r, r.vecTable(tVec, i)))
        }
        val inputs = r.fieldPos(pos, 6).let { if (it != 0L) r.intVec(it) else null }
        val outputs = r.fieldPos(pos, 8).let { if (it != 0L) r.intVec(it) else null }
        val ops = ArrayList<OpSpec>()
        val oVec = r.fieldPos(pos, 10)
        if (oVec != 0L) {
            val n = r.vecLen(oVec)
            for (i in 0 until n) ops.add(readOp(r, r.vecTable(oVec, i)))
        }
        val name = r.fieldPos(pos, 12).let { if (it != 0L) r.str(it) else null }
        val dbgPos = r.fieldPos(pos, 14)
        val dbg = if (dbgPos != 0L) r.i32(dbgPos) else -1
        return SubGraphSpec(tensors, inputs, outputs, ops, name, dbg, dbgPos != 0L)
    }

    private fun readTensor(r: FbReader, pos: Long): TensorSpec {
        check(r.fieldPos(pos, 16) == 0L && r.fieldPos(pos, 22) == 0L) {
            "Unsupported tensor sparsity/variant_tensors in TFLite model"
        }
        val shape = r.fieldPos(pos, 4).let { if (it != 0L) r.intVec(it) else null }
        val typePos = r.fieldPos(pos, 6)
        val type = if (typePos != 0L) r.i8(typePos) else 0
        val bufPos = r.fieldPos(pos, 8)
        val buf = if (bufPos != 0L) r.i32(bufPos) else 0
        val name = r.fieldPos(pos, 10).let { if (it != 0L) r.str(it) else null }
        val quant = r.fieldPos(pos, 12).let { if (it != 0L) readQuant(r, r.indirect(it)) else null }
        val varPos = r.fieldPos(pos, 14)
        val isVar = varPos != 0L && r.i8(varPos) != 0.toByte()
        val sig = r.fieldPos(pos, 18).let { if (it != 0L) r.intVec(it) else null }
        val hrPos = r.fieldPos(pos, 20)
        val hasRank = hrPos != 0L && r.i8(hrPos) != 0.toByte()
        val ebPos = r.fieldPos(pos, 24)
        val extBuf = if (ebPos != 0L) r.i32(ebPos) else 0
        return TensorSpec(
            shape, type, typePos != 0L, buf, bufPos != 0L, name, quant,
            isVar, varPos != 0L, sig, hasRank, hrPos != 0L, extBuf, ebPos != 0L,
        )
    }

    private fun readQuant(r: FbReader, pos: Long): QuantSpec {
        val min = r.fieldPos(pos, 4).let { if (it != 0L) r.floatVec(it) else null }
        val max = r.fieldPos(pos, 6).let { if (it != 0L) r.floatVec(it) else null }
        val scale = r.fieldPos(pos, 8).let { if (it != 0L) r.floatVec(it) else null }
        val zp = r.fieldPos(pos, 10).let { if (it != 0L) r.longVec(it) else null }
        val dt = r.fieldPos(pos, 12).let { if (it != 0L) r.u8(it) else 0 }
        val dp = r.fieldPos(pos, 14).let { if (it != 0L) r.indirect(it) else 0L }
        check(dt == 0) { "Unsupported QuantizationDetails type $dt" }
        val qdPos = r.fieldPos(pos, 16)
        val qd = if (qdPos != 0L) r.i32(qdPos) else 0
        return QuantSpec(min, max, scale, zp, dt, dp, qd, qdPos != 0L)
    }

    private fun readOp(r: FbReader, pos: Long): OpSpec {
        val oc = r.fieldPos(pos, 4).let { if (it != 0L) r.i32(it) else 0 }
        val ins = r.fieldPos(pos, 6).let { if (it != 0L) r.intVec(it) else null }
        val outs = r.fieldPos(pos, 8).let { if (it != 0L) r.intVec(it) else null }
        val boType = r.fieldPos(pos, 10).let { if (it != 0L) r.u8(it) else 0 }
        val boPos = r.fieldPos(pos, 12).let { if (it != 0L) r.indirect(it) else 0L }
        val co = r.fieldPos(pos, 14).let { if (it != 0L) r.byteVec(it) else null }
        val cofPos = r.fieldPos(pos, 16)
        val cof = if (cofPos != 0L) r.i8(cofPos) else 0
        val mvi = r.fieldPos(pos, 18).let { if (it != 0L) r.byteVec(it) else null }
        val interms = r.fieldPos(pos, 20).let { if (it != 0L) r.intVec(it) else null }
        val lcoOff = r.fieldPos(pos, 22).let { if (it != 0L) r.i64(it) else 0L }
        val lcoSz = r.fieldPos(pos, 24).let { if (it != 0L) r.i64(it) else 0L }
        val bo2Type = r.fieldPos(pos, 26).let { if (it != 0L) r.u8(it) else 0 }
        val bo2Pos = r.fieldPos(pos, 28).let { if (it != 0L) r.indirect(it) else 0L }
        check(bo2Type == 0) { "Unsupported BuiltinOptions2 type $bo2Type" }
        val dbgPos = r.fieldPos(pos, 30)
        val dbg = if (dbgPos != 0L) r.i32(dbgPos) else -1
        return OpSpec(
            oc, ins, outs, boType, boPos, co, cof, cofPos != 0L, mvi, interms,
            lcoOff, lcoSz, bo2Type, bo2Pos, dbg, dbgPos != 0L,
        )
    }

    private fun readBuffer(r: FbReader, pos: Long): BufferSpec {
        val dField = r.fieldPos(pos, 4)
        val dataLen = if (dField != 0L) r.vecLen(dField) else 0
        val dataPos = if (dField != 0L) r.indirect(dField) + 4L else 0L
        val off = r.fieldPos(pos, 6).let { if (it != 0L) r.i64(it) else 0L }
        val sz = r.fieldPos(pos, 8).let { if (it != 0L) r.i64(it) else 0L }
        return BufferSpec(dataPos, dataLen, off, sz)
    }

    private fun readSignatureDef(r: FbReader, pos: Long): SignatureDefSpec {
        fun readMaps(fieldOffset: Int): List<TensorMapSpec> {
            val v = r.fieldPos(pos, fieldOffset)
            if (v == 0L) return emptyList()
            val n = r.vecLen(v)
            return List(n) { i ->
                val p = r.vecTable(v, i)
                val name = r.fieldPos(p, 4).let { if (it != 0L) r.str(it) else null }
                val ti = r.fieldPos(p, 6).let { if (it != 0L) r.i32(it) else 0 }
                TensorMapSpec(name, ti)
            }
        }
        val key = r.fieldPos(pos, 8).let { if (it != 0L) r.str(it) else null }
        val sgPos = r.fieldPos(pos, 12)
        val sgIdx = if (sgPos != 0L) r.i32(sgPos) else 0
        return SignatureDefSpec(readMaps(4), readMaps(6), key, sgIdx, sgPos != 0L)
    }

    private fun writeModelHeader(
        r: FbReader,
        model: ModelSpec,
        isExternalBuf: BooleanArray,
        inlineData: Array<ByteArray?>,
        bufTableOffsetsOut: IntArray,
    ): ByteArray {
        val fb = FbWriter()

        val ocOffsets = model.operatorCodes.map { oc ->
            val customOff = oc.customCode?.let { fb.writeString(it) } ?: 0
            fb.startTable(4)
            if (oc.deprecatedBuiltinCode != 0.toByte()) fb.addByte(0, oc.deprecatedBuiltinCode)
            if (customOff != 0) fb.addOffset(1, customOff)
            if (oc.hasVersion) fb.addInt(2, oc.version)
            if (oc.hasBuiltinCode) fb.addInt(3, oc.builtinCode)
            fb.endTable()
        }.toIntArray()
        val ocVecOff = fb.writeOffsetVector(ocOffsets)

        val sgOffsets = model.subgraphs.map { sg ->
            val tOffsets = sg.tensors.map { t ->
                val shapeOff = t.shape?.let { fb.writeIntVector(it) } ?: 0
                val nameOff = t.name?.let { fb.writeString(it) } ?: 0
                val quantOff = t.quantization?.let { q ->
                    val minOff = q.min?.let { fb.writeFloatVector(it) } ?: 0
                    val maxOff = q.max?.let { fb.writeFloatVector(it) } ?: 0
                    val scaleOff = q.scale?.let { fb.writeFloatVector(it) } ?: 0
                    val zpOff = q.zeroPoint?.let { fb.writeLongVector(it) } ?: 0
                    fb.startTable(7)
                    if (minOff != 0) fb.addOffset(0, minOff)
                    if (maxOff != 0) fb.addOffset(1, maxOff)
                    if (scaleOff != 0) fb.addOffset(2, scaleOff)
                    if (zpOff != 0) fb.addOffset(3, zpOff)
                    if (q.detailsType != 0) fb.addByte(4, q.detailsType.toByte())
                    if (q.hasQuantizedDimension) fb.addInt(6, q.quantizedDimension)
                    fb.endTable()
                } ?: 0
                val sigOff = t.shapeSignature?.let { fb.writeIntVector(it) } ?: 0
                fb.startTable(11)
                if (shapeOff != 0) fb.addOffset(0, shapeOff)
                if (t.hasType) fb.addByte(1, t.type)
                if (t.hasBuffer) fb.addInt(2, t.buffer)
                if (nameOff != 0) fb.addOffset(3, nameOff)
                if (quantOff != 0) fb.addOffset(4, quantOff)
                if (t.hasIsVariable) fb.addByte(5, if (t.isVariable) 1 else 0)
                if (sigOff != 0) fb.addOffset(7, sigOff)
                if (t.hasHasRank) fb.addByte(8, if (t.hasRank) 1 else 0)
                if (t.hasExternalBuffer) fb.addInt(10, t.externalBuffer)
                fb.endTable()
            }.toIntArray()
            val tensorsVec = fb.writeOffsetVector(tOffsets)
            val inputsVec = sg.inputs?.let { fb.writeIntVector(it) } ?: 0
            val outputsVec = sg.outputs?.let { fb.writeIntVector(it) } ?: 0
            val opOffsets = sg.operators.map { op ->
                val inOff = op.inputs?.let { fb.writeIntVector(it) } ?: 0
                val outOff = op.outputs?.let { fb.writeIntVector(it) } ?: 0
                val boOff = writeBuiltinOptions(r, op.builtinOptionsType, op.builtinOptionsPos, fb)
                val coOff = op.customOptions?.let { fb.writeByteVector(it, 1) } ?: 0
                val mviOff = op.mutatingVariableInputs?.let { fb.writeByteVector(it, 1) } ?: 0
                val intOff = op.intermediates?.let { fb.writeIntVector(it) } ?: 0
                fb.startTable(14)
                if (op.opcodeIndex != 0) fb.addInt(0, op.opcodeIndex)
                if (inOff != 0) fb.addOffset(1, inOff)
                if (outOff != 0) fb.addOffset(2, outOff)
                if (op.builtinOptionsType != 0) fb.addByte(3, op.builtinOptionsType.toByte())
                if (boOff != 0) fb.addOffset(4, boOff)
                if (coOff != 0) fb.addOffset(5, coOff)
                if (op.hasCustomOptionsFormat) fb.addByte(6, op.customOptionsFormat)
                if (mviOff != 0) fb.addOffset(7, mviOff)
                if (intOff != 0) fb.addOffset(8, intOff)
                if (op.largeCustomOptionsOffset != 0L) fb.addLong(9, op.largeCustomOptionsOffset)
                if (op.largeCustomOptionsSize != 0L) fb.addLong(10, op.largeCustomOptionsSize)
                if (op.hasDebugMetadataIndex) fb.addInt(13, op.debugMetadataIndex)
                fb.endTable()
            }.toIntArray()
            val opsVec = fb.writeOffsetVector(opOffsets)
            val sgNameOff = sg.name?.let { fb.writeString(it) } ?: 0
            fb.startTable(6)
            fb.addOffset(0, tensorsVec)
            if (inputsVec != 0) fb.addOffset(1, inputsVec)
            if (outputsVec != 0) fb.addOffset(2, outputsVec)
            fb.addOffset(3, opsVec)
            if (sgNameOff != 0) fb.addOffset(4, sgNameOff)
            if (sg.hasDebugMetadataIndex) fb.addInt(5, sg.debugMetadataIndex)
            fb.endTable()
        }.toIntArray()
        val sgVecOff = fb.writeOffsetVector(sgOffsets)

        val descOff = model.description?.let { fb.writeString(it) } ?: 0

        val bufOffsets = IntArray(model.buffers.size)
        for (i in model.buffers.indices) {
            val dataVecOff = inlineData[i]?.let { fb.writeByteVector(it, 16) } ?: 0
            fb.startTable(3)
            if (dataVecOff != 0) fb.addOffset(0, dataVecOff)
            if (isExternalBuf[i]) {
                fb.addLong(1, 1L)
                fb.addLong(2, 1L)
            }
            val off = fb.endTable()
            bufOffsets[i] = off
            bufTableOffsetsOut[i] = off
        }
        val bufVecOff = fb.writeOffsetVector(bufOffsets)

        val mbVecOff = model.metadataBuffer?.let { fb.writeIntVector(it) } ?: 0

        val mdOffsets = model.metadata.map { md ->
            val nOff = md.name?.let { fb.writeString(it) } ?: 0
            fb.startTable(2)
            if (nOff != 0) fb.addOffset(0, nOff)
            if (md.buffer != 0) fb.addInt(1, md.buffer)
            fb.endTable()
        }.toIntArray()
        val mdVecOff = if (mdOffsets.isNotEmpty()) fb.writeOffsetVector(mdOffsets) else 0

        val sdOffsets = model.signatureDefs.map { sd ->
            fun writeMaps(maps: List<TensorMapSpec>): Int {
                if (maps.isEmpty()) return 0
                val offs = maps.map { tm ->
                    val nOff = tm.name?.let { fb.writeString(it) } ?: 0
                    fb.startTable(2)
                    if (nOff != 0) fb.addOffset(0, nOff)
                    if (tm.tensorIndex != 0) fb.addInt(1, tm.tensorIndex)
                    fb.endTable()
                }.toIntArray()
                return fb.writeOffsetVector(offs)
            }
            val inOff = writeMaps(sd.inputs)
            val outOff = writeMaps(sd.outputs)
            val keyOff = sd.signatureKey?.let { fb.writeString(it) } ?: 0
            fb.startTable(5)
            if (inOff != 0) fb.addOffset(0, inOff)
            if (outOff != 0) fb.addOffset(1, outOff)
            if (keyOff != 0) fb.addOffset(2, keyOff)
            if (sd.hasSubgraphIndex) fb.addInt(4, sd.subgraphIndex)
            fb.endTable()
        }.toIntArray()
        val sdVecOff = if (sdOffsets.isNotEmpty()) fb.writeOffsetVector(sdOffsets) else 0

        fb.startTable(8)
        if (model.version != 0) fb.addInt(0, model.version)
        fb.addOffset(1, ocVecOff)
        fb.addOffset(2, sgVecOff)
        if (descOff != 0) fb.addOffset(3, descOff)
        fb.addOffset(4, bufVecOff)
        if (mbVecOff != 0) fb.addOffset(5, mbVecOff)
        if (mdVecOff != 0) fb.addOffset(6, mdVecOff)
        if (sdVecOff != 0) fb.addOffset(7, sdVecOff)
        val root = fb.endTable()
        return fb.finish(root, "TFL3")
    }

    private fun writeBuiltinOptions(r: FbReader, type: Int, pos: Long, fb: FbWriter): Int {
        if (type == BO_NONE || pos == 0L) return 0
        return when (type) {
            BO_FULLY_CONNECTED -> {
                val faf = r.fieldPos(pos, 4)
                val wf = r.fieldPos(pos, 6)
                val knd = r.fieldPos(pos, 8)
                val aqi = r.fieldPos(pos, 10)
                val qbt = r.fieldPos(pos, 12)
                val qs = r.fieldPos(pos, 14).let { if (it != 0L) r.byteVec(it) else null }
                val qsOff = qs?.let { fb.writeByteVector(it, 1) } ?: 0
                fb.startTable(6)
                if (faf != 0L) fb.addByte(0, r.i8(faf))
                if (wf != 0L) fb.addByte(1, r.i8(wf))
                if (knd != 0L) fb.addByte(2, r.i8(knd))
                if (aqi != 0L) fb.addByte(3, r.i8(aqi))
                if (qbt != 0L) fb.addByte(4, r.i8(qbt))
                if (qsOff != 0) fb.addOffset(5, qsOff)
                fb.endTable()
            }
            BO_RESHAPE -> {
                val ns = r.fieldPos(pos, 4).let { if (it != 0L) r.intVec(it) else null }
                val nsOff = ns?.let { fb.writeIntVector(it) } ?: 0
                fb.startTable(1)
                if (nsOff != 0) fb.addOffset(0, nsOff)
                fb.endTable()
            }
            BO_SQUEEZE -> {
                val sd = r.fieldPos(pos, 4).let { if (it != 0L) r.intVec(it) else null }
                val sdOff = sd?.let { fb.writeIntVector(it) } ?: 0
                fb.startTable(1)
                if (sdOff != 0) fb.addOffset(0, sdOff)
                fb.endTable()
            }
            BO_CONCAT_EMBEDDINGS, BO_VAR_HANDLE, BO_BUCKETIZE -> {
                error("Unsupported BuiltinOptions with offset fields: $type")
            }
            else -> copyScalarTable(r, pos, fb)
        }
    }

    /** Copies any FlatBuffer table whose fields are purely inline scalars verbatim. */
    private fun copyScalarTable(r: FbReader, srcPos: Long, fb: FbWriter): Int {
        val vtablePos = srcPos - r.i32(srcPos)
        val vtableSize = r.u16(vtablePos)
        val objectSize = r.u16(vtablePos + 2)
        val slots = (vtableSize - 4) / 2
        val vtableSlots = IntArray(slots) { i -> r.u16(vtablePos + 4 + i * 2) }
        val payload = if (objectSize > 4) r.bytes(srcPos + 4, objectSize - 4) else ByteArray(0)
        return fb.writeRawTable(payload, objectSize, vtableSlots)
    }

    /** Buffered random-access FlatBuffer reader using 64-bit file offsets. */
    private class FbReader(private val raf: RandomAccessFile) {
        private val fileLen = raf.length()
        private val cache = ByteArray(1 shl 20)
        private var cacheStart = -1L
        private var cacheLen = 0

        private fun ensureWindow(pos: Long, len: Int) {
            require(pos >= 0 && pos + len <= fileLen) { "Read out of bounds: $pos+$len > $fileLen" }
            if (cacheStart >= 0 && pos >= cacheStart && pos + len <= cacheStart + cacheLen) return
            if (len > cache.size) return
            cacheStart = pos
            val want = minOf(cache.size.toLong(), fileLen - pos).toInt()
            raf.seek(pos)
            raf.readFully(cache, 0, want)
            cacheLen = want
        }

        fun bytes(pos: Long, len: Int): ByteArray {
            if (len == 0) return ByteArray(0)
            if (len <= cache.size) {
                ensureWindow(pos, len)
                val off = (pos - cacheStart).toInt()
                return cache.copyOfRange(off, off + len)
            }
            val out = ByteArray(len)
            raf.seek(pos)
            raf.readFully(out)
            return out
        }

        fun i8(pos: Long): Byte {
            ensureWindow(pos, 1)
            return cache[(pos - cacheStart).toInt()]
        }

        fun u8(pos: Long): Int = i8(pos).toInt() and 0xff

        fun u16(pos: Long): Int {
            ensureWindow(pos, 2)
            val o = (pos - cacheStart).toInt()
            return (cache[o].toInt() and 0xff) or ((cache[o + 1].toInt() and 0xff) shl 8)
        }

        fun i32(pos: Long): Int {
            ensureWindow(pos, 4)
            val o = (pos - cacheStart).toInt()
            return (cache[o].toInt() and 0xff) or
                ((cache[o + 1].toInt() and 0xff) shl 8) or
                ((cache[o + 2].toInt() and 0xff) shl 16) or
                ((cache[o + 3].toInt() and 0xff) shl 24)
        }

        fun u32(pos: Long): Long = i32(pos).toLong() and 0xffffffffL

        fun i64(pos: Long): Long {
            ensureWindow(pos, 8)
            val o = (pos - cacheStart).toInt()
            var v = 0L
            for (k in 0 until 8) v = v or ((cache[o + k].toLong() and 0xffL) shl (k * 8))
            return v
        }

        fun fieldPos(tablePos: Long, vtableByteOffset: Int): Long {
            val vtablePos = tablePos - i32(tablePos)
            val vtableSize = u16(vtablePos)
            if (vtableByteOffset >= vtableSize) return 0L
            val rel = u16(vtablePos + vtableByteOffset)
            return if (rel == 0) 0L else tablePos + rel
        }

        fun indirect(pos: Long): Long = pos + u32(pos)

        fun vecLen(fieldPos: Long): Int = i32(indirect(fieldPos))

        fun vecTable(fieldPos: Long, idx: Int): Long = indirect(indirect(fieldPos) + 4L + idx.toLong() * 4L)

        fun str(fieldPos: Long): String {
            val p = indirect(fieldPos)
            val len = i32(p)
            return String(bytes(p + 4L, len), Charsets.UTF_8)
        }

        fun byteVec(fieldPos: Long): ByteArray {
            val p = indirect(fieldPos)
            val len = i32(p)
            return bytes(p + 4L, len)
        }

        fun intVec(fieldPos: Long): IntArray {
            val p = indirect(fieldPos)
            val len = i32(p)
            val raw = bytes(p + 4L, len * 4)
            val out = IntArray(len)
            ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN).asIntBuffer().get(out)
            return out
        }

        fun floatVec(fieldPos: Long): FloatArray {
            val p = indirect(fieldPos)
            val len = i32(p)
            val raw = bytes(p + 4L, len * 4)
            val out = FloatArray(len)
            ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(out)
            return out
        }

        fun longVec(fieldPos: Long): LongArray {
            val p = indirect(fieldPos)
            val len = i32(p)
            val raw = bytes(p + 4L, len * 8)
            val out = LongArray(len)
            ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN).asLongBuffer().get(out)
            return out
        }
    }

    /** Minimal bottom-up FlatBuffer serializer. */
    private class FbWriter {
        private var buf = ByteArray(64 * 1024)
        private var space = buf.size

        private var inTable = false
        private var tableBottom = 0
        private var currentVtable = IntArray(0)
        private var currentNumFields = 0
        private val vtableCache = HashMap<VTableKey, Int>()

        private class VTableKey(val objSize: Int, val slots: IntArray) {
            override fun equals(other: Any?): Boolean =
                other is VTableKey && objSize == other.objSize && slots.contentEquals(other.slots)
            override fun hashCode(): Int = 31 * objSize + slots.contentHashCode()
        }

        fun offset(): Int = buf.size - space

        private fun ensure(needed: Int) {
            if (space >= needed) return
            var cap = buf.size
            val used = offset()
            while (cap - used < needed) cap = cap shl 1
            val next = ByteArray(cap)
            val newSpace = cap - used
            System.arraycopy(buf, space, next, newSpace, used)
            buf = next
            space = newSpace
        }

        private fun prep(align: Int, additionalBytes: Int) {
            val misalign = (offset() + additionalBytes) % align
            val pad = if (misalign == 0) 0 else align - misalign
            ensure(pad)
            for (i in 0 until pad) buf[--space] = 0
        }

        private fun putByte(v: Byte) {
            ensure(1)
            buf[--space] = v
        }

        private fun putShort(v: Short) {
            prep(2, 0)
            ensure(2)
            val x = v.toInt()
            buf[--space] = (x ushr 8).toByte()
            buf[--space] = x.toByte()
        }

        private fun putInt(v: Int) {
            prep(4, 0)
            ensure(4)
            buf[--space] = (v ushr 24).toByte()
            buf[--space] = (v ushr 16).toByte()
            buf[--space] = (v ushr 8).toByte()
            buf[--space] = v.toByte()
        }

        private fun putLong(v: Long) {
            prep(8, 0)
            ensure(8)
            for (k in 7 downTo 0) buf[--space] = (v ushr (k * 8)).toByte()
        }

        private fun putBytes(src: ByteArray) {
            ensure(src.size)
            space -= src.size
            System.arraycopy(src, 0, buf, space, src.size)
        }

        private fun putOffset(targetOffset: Int) {
            prep(4, 0)
            val rel = (offset() + 4) - targetOffset
            putInt(rel)
        }

        fun writeString(s: String): Int {
            val bytes = s.toByteArray(Charsets.UTF_8)
            prep(4, 1 + bytes.size)
            putByte(0)
            putBytes(bytes)
            putInt(bytes.size)
            return offset()
        }

        fun writeByteVector(bytes: ByteArray, align: Int): Int {
            prep(maxOf(4, align), bytes.size)
            putBytes(bytes)
            putInt(bytes.size)
            return offset()
        }

        fun writeIntVector(values: IntArray): Int {
            prep(4, values.size * 4)
            for (i in values.size - 1 downTo 0) putInt(values[i])
            putInt(values.size)
            return offset()
        }

        fun writeFloatVector(values: FloatArray): Int {
            prep(4, values.size * 4)
            for (i in values.size - 1 downTo 0) putInt(java.lang.Float.floatToRawIntBits(values[i]))
            putInt(values.size)
            return offset()
        }

        fun writeLongVector(values: LongArray): Int {
            prep(8, values.size * 8)
            for (i in values.size - 1 downTo 0) putLong(values[i])
            putInt(values.size)
            return offset()
        }

        fun writeOffsetVector(targets: IntArray): Int {
            prep(4, targets.size * 4)
            for (i in targets.size - 1 downTo 0) putOffset(targets[i])
            putInt(targets.size)
            return offset()
        }

        fun startTable(numFields: Int) {
            check(!inTable)
            inTable = true
            if (currentVtable.size < numFields) currentVtable = IntArray(numFields)
            for (i in 0 until numFields) currentVtable[i] = 0
            currentNumFields = numFields
            tableBottom = offset()
        }

        fun addByte(idx: Int, v: Byte) {
            putByte(v)
            currentVtable[idx] = offset()
        }

        fun addInt(idx: Int, v: Int) {
            putInt(v)
            currentVtable[idx] = offset()
        }

        fun addLong(idx: Int, v: Long) {
            putLong(v)
            currentVtable[idx] = offset()
        }

        fun addOffset(idx: Int, target: Int) {
            putOffset(target)
            currentVtable[idx] = offset()
        }

        fun endTable(): Int {
            check(inTable)
            inTable = false
            putInt(0) // placeholder for soffset_t to vtable
            val tableStart = offset()
            val objSize = tableStart - tableBottom
            var trimmedFields = currentNumFields
            while (trimmedFields > 0 && currentVtable[trimmedFields - 1] == 0) trimmedFields--
            val relSlots = IntArray(trimmedFields) { i ->
                if (currentVtable[i] == 0) 0 else tableStart - currentVtable[i]
            }
            val key = VTableKey(objSize, relSlots)
            val vtableOffset = vtableCache[key] ?: run {
                for (i in trimmedFields - 1 downTo 0) putShort(relSlots[i].toShort())
                putShort(objSize.toShort())
                putShort((4 + trimmedFields * 2).toShort())
                val vo = offset()
                vtableCache[key] = vo
                vo
            }
            // Write soffset_t (tablePos - vtablePos == vtableOffset - tableStart) at tableStart.
            val soffPos = buf.size - tableStart
            val soff = vtableOffset - tableStart
            buf[soffPos] = soff.toByte()
            buf[soffPos + 1] = (soff ushr 8).toByte()
            buf[soffPos + 2] = (soff ushr 16).toByte()
            buf[soffPos + 3] = (soff ushr 24).toByte()
            return tableStart
        }

        fun writeRawTable(payload: ByteArray, objectSize: Int, slots: IntArray): Int {
            check(!inTable)
            prep(8, objectSize)
            if (payload.isNotEmpty()) putBytes(payload)
            putInt(0)
            val tableStart = offset()
            val key = VTableKey(objectSize, slots)
            val vtableOffset = vtableCache[key] ?: run {
                for (i in slots.size - 1 downTo 0) putShort(slots[i].toShort())
                putShort(objectSize.toShort())
                putShort((4 + slots.size * 2).toShort())
                val vo = offset()
                vtableCache[key] = vo
                vo
            }
            val soffPos = buf.size - tableStart
            val soff = vtableOffset - tableStart
            buf[soffPos] = soff.toByte()
            buf[soffPos + 1] = (soff ushr 8).toByte()
            buf[soffPos + 2] = (soff ushr 16).toByte()
            buf[soffPos + 3] = (soff ushr 24).toByte()
            return tableStart
        }

        fun finish(rootTableOffset: Int, fileIdentifier: String): ByteArray {
            val id = fileIdentifier.toByteArray(Charsets.US_ASCII)
            check(id.size == 4)
            prep(16, 4 + 4)
            putBytes(id)
            putOffset(rootTableOffset)
            return buf.copyOfRange(space, buf.size)
        }
    }
}
