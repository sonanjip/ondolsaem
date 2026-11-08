package io.github.taehunkim.ondolsaem.engine.iris

import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteOrder
import java.nio.channels.FileChannel

/**
 * The host-side tensor work of the iris-3b pipeline: everything between the graphs that is not a
 * learned weight.
 */
object HostOps {
    const val SEQ = 334
    const val PREFIX = 34
    const val WINDOW = 300
    const val TE_HIDDEN = 2560
    const val TE_HEADS = 32
    const val TAPS = 12
    const val ADAPTER_HEADS = 20
    const val DIT_HIDDEN = 2560
    const val IMAGE_SIZE = 768
    const val PATCH = 16
    const val GRID = IMAGE_SIZE / PATCH
    const val PATCHES = GRID * GRID
    const val PIXELS = PATCH * PATCH
    const val CHANNELS = 3
    const val IMAGE_FLOATS = CHANNELS * IMAGE_SIZE * IMAGE_SIZE
    const val TE_FLOATS = TAPS * WINDOW * TE_HIDDEN
    const val MASK_NEG = -30000f

    /**
     * The text encoder's additive attention mask [1, 32, 334, 334]: 0 where key <= query and the
     * key is a real token, else [MASK_NEG].
     */
    fun teMask(realLen: Int): FloatArray {
        val plane = FloatArray(SEQ * SEQ)
        for (query in 0 until SEQ) {
            for (key in 0 until SEQ) {
                plane[query * SEQ + key] = if (key <= query && key < realLen) 0f else MASK_NEG
            }
        }
        return repeatPlane(plane, TE_HEADS)
    }

    /** The 300-row window's mask: 1 on caption + suffix rows. */
    fun windowMask(realLen: Int): FloatArray =
        FloatArray(WINDOW) { if (PREFIX + it < realLen) 1f else 0f }

    /**
     * Writes tap [tap]'s window rows times the window mask into the adapter input [te].
     */
    fun packTap(tap: Int, hidden: FloatArray, windowMask: FloatArray, te: FloatArray) {
        for (row in 0 until WINDOW) {
            val src = (PREFIX + row) * TE_HIDDEN
            val dst = (row * TAPS + tap) * TE_HIDDEN
            val m = windowMask[row]
            for (c in 0 until TE_HIDDEN) {
                te[dst + c] = hidden[src + c] * m
            }
        }
    }

    /**
     * The adapter refiner's additive key mask [1, 20, 300, 300].
     */
    fun adapterMaskBias(windowMask: FloatArray): FloatArray {
        val plane = FloatArray(WINDOW * WINDOW)
        for (query in 0 until WINDOW) {
            for (key in 0 until WINDOW) {
                plane[query * WINDOW + key] = if (windowMask[key] != 0f || query == key) 0f else MASK_NEG
            }
        }
        return repeatPlane(plane, ADAPTER_HEADS)
    }

    /** Rows [first, end) of a [rows, width] table. */
    fun rows(table: FloatArray, width: Int, first: Int, end: Int): FloatArray =
        table.copyOfRange(first * width, end * width)

    /** The single-stream blocks' joint sequence: text rows first, then image rows. */
    fun concatTextImage(text: FloatArray, image: FloatArray): FloatArray {
        val out = FloatArray(text.size + image.size)
        System.arraycopy(text, 0, out, 0, text.size)
        System.arraycopy(image, 0, out, text.size, image.size)
        return out
    }

    /** joint[:, 300:] = the image rows the pixel stage is conditioned on. */
    fun imageRows(joint: FloatArray): FloatArray = joint.copyOfRange(WINDOW * DIT_HIDDEN, joint.size)

    /**
     * One step of the Turbo few-step sampler in place: clean = x - sigma * v; when there is a next sigma s',
     * x' = (1 - s') * clean + s' * (x - (1 - sigma) * clean) / sigma.
     */
    fun samplerStep(x: FloatArray, v: FloatArray, sigma: Float, next: Float?, clean: FloatArray) {
        for (i in x.indices) {
            clean[i] = x[i] - sigma * v[i]
        }
        if (next == null) {
            return
        }
        val keep = 1f - sigma
        val keepNext = 1f - next
        for (i in x.indices) {
            val implied = (x[i] - keep * clean[i]) / sigma
            x[i] = keepNext * clean[i] + next * implied
        }
    }

    /** Clamps to [-1, 1] in place. */
    fun clampUnit(values: FloatArray) {
        for (i in values.indices) {
            val v = values[i]
            if (v < -1f) values[i] = -1f else if (v > 1f) values[i] = 1f
        }
    }

    /** The patchify map. */
    fun patchIndex(): IntArray {
        val out = IntArray(IMAGE_FLOATS)
        var k = 0
        for (gy in 0 until GRID) {
            for (gx in 0 until GRID) {
                for (py in 0 until PATCH) {
                    for (px in 0 until PATCH) {
                        for (c in 0 until CHANNELS) {
                            out[k++] = c * IMAGE_SIZE * IMAGE_SIZE + (gy * PATCH + py) * IMAGE_SIZE + (gx * PATCH + px)
                        }
                    }
                }
            }
        }
        return out
    }

    /** Patch-major [2304, 256, 3] -> CHW [3, 768, 768] through [index]. */
    fun toImage(xPix: FloatArray, index: IntArray): FloatArray {
        val out = FloatArray(IMAGE_FLOATS)
        for (k in index.indices) {
            out[index[k]] = xPix[k]
        }
        return out
    }

    /** Count of NaN / infinite values. */
    fun nonFinite(values: FloatArray): Int = values.count { !it.isFinite() }

    /** Little-endian float32 values of a file region. */
    fun readFloats(file: File, offset: Long, count: Int): FloatArray {
        val out = FloatArray(count)
        RandomAccessFile(file, "r").use { raf ->
            val map = raf.channel.map(FileChannel.MapMode.READ_ONLY, offset, count * 4L)
            map.order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(out)
        }
        return out
    }

    /** Little-endian int32 values of a whole file. */
    fun readInts(file: File): IntArray {
        val count = (file.length() / 4).toInt()
        val out = IntArray(count)
        RandomAccessFile(file, "r").use { raf ->
            val map = raf.channel.map(FileChannel.MapMode.READ_ONLY, 0, count * 4L)
            map.order(ByteOrder.LITTLE_ENDIAN).asIntBuffer().get(out)
        }
        return out
    }

    private fun repeatPlane(plane: FloatArray, heads: Int): FloatArray {
        val out = FloatArray(heads * plane.size)
        for (head in 0 until heads) {
            System.arraycopy(plane, 0, out, head * plane.size, plane.size)
        }
        return out
    }
}
