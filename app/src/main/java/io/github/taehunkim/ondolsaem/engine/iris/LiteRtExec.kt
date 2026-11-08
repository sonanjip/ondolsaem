package io.github.taehunkim.ondolsaem.engine.iris

import android.os.Build
import android.util.Log
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.Environment
import com.google.ai.edge.litert.TensorBuffer
import java.io.File
import java.util.Collections

/**
 * [GraphExec] on LiteRT CompiledModel, one accelerator per graph as its [GraphSpec] says: `gpu` =
 * the GPU, `cpu` = XNNPACK.
 */
class LiteRtExec(
    private val env: Environment,
    private val dir: File,
    private val programCache: File? = null,
    private val cacheRoles: List<String> = emptyList(),
) : GraphExec {

    val cacheDecisions: MutableMap<String, String> = Collections.synchronizedMap(LinkedHashMap<String, String>())

    private class Loaded(
        override val spec: GraphSpec,
        val model: CompiledModel,
        val inputs: Map<String, TensorBuffer>,
        val outputs: Map<String, TensorBuffer>,
    ) : LoadedGraph

    private val isEmulator: Boolean
        get() = Build.FINGERPRINT.contains("generic") || Build.HARDWARE.contains("ranchu") || Build.HARDWARE.contains("goldfish")

    override fun load(spec: GraphSpec): LoadedGraph {
        val graphFile = File(dir, spec.file)
        val accel = if (isEmulator) "cpu" else spec.accel
        val options = if (accel == "cpu") {
            val weightCache = xnnpackCacheFile(dir, spec).takeIf {
                spec.xnnpackCache && roomFor("${spec.role} xnnpack", it, graphFile)
            }
            CompiledModel.Options(Accelerator.CPU).apply {
                cpuOptions = CompiledModel.CpuOptions(
                    numThreads = spec.threads,
                    xnnPackWeightCachePath = weightCache?.absolutePath,
                )
            }
        } else {
            val cache = programCache?.takeIf { cacheAllowed(spec, graphFile, it) }
            CompiledModel.Options(Accelerator.GPU).apply {
                gpuOptions = CompiledModel.GpuOptions(
                    precision = precisionOf(spec.precision),
                    serializationDir = cache?.absolutePath,
                    modelCacheKey = cache?.let { cacheKey(spec) },
                    serializeProgramCache = cache?.let { true },
                )
            }
        }
        val model = try {
            CompiledModel.create(graphFile.absolutePath, options, env)
        } catch (e: Throwable) {
            if (accel == "gpu") {
                Log.w("LiteRtExec", "GPU compile failed for ${spec.file}, falling back to CPU", e)
                val cpuOpts = CompiledModel.Options(Accelerator.CPU).apply {
                    cpuOptions = CompiledModel.CpuOptions(numThreads = spec.threads)
                }
                CompiledModel.create(graphFile.absolutePath, cpuOpts, env)
            } else throw e
        }
        val buffers = ArrayList<TensorBuffer>()
        try {
            val inputs = spec.inputs.associateWith { model.createInputBuffer(it, spec.signature).also(buffers::add) }
            val outputs = spec.outputs.associate { (name, _) -> name to model.createOutputBuffer(name, spec.signature).also(buffers::add) }
            return Loaded(spec, model, inputs, outputs)
        } catch (failure: Throwable) {
            buffers.forEach { runCatching { it.close() } }
            runCatching { model.close() }
            throw failure
        }
    }

    override fun run(graph: LoadedGraph, inputs: List<FloatArray>): GraphRun {
        val g = graph as Loaded
        require(inputs.size == g.spec.inputs.size) { "${g.spec.role}: ${inputs.size} inputs for ${g.spec.inputs.size} names" }
        val t0 = System.nanoTime()
        for ((i, name) in g.spec.inputs.withIndex()) {
            val buf = g.inputs.getValue(name)
            buf.writeFloat(inputs[i])
        }
        val t1 = System.nanoTime()
        g.model.run(g.inputs, g.outputs, g.spec.signature)
        val t2 = System.nanoTime()
        val outs = g.spec.outputs.map { (name, count) ->
            val buf = g.outputs.getValue(name)
            val values = buf.readFloat()
            require(values.size == count) { "${g.spec.role} $name: ${values.size} floats, the config says $count" }
            values
        }
        val t3 = System.nanoTime()
        return GraphRun(outs, t1 - t0, t2 - t1, t3 - t2)
    }

    override fun close(graph: LoadedGraph) {
        val g = graph as Loaded
        g.inputs.values.forEach { runCatching { it.close() } }
        g.outputs.values.forEach { runCatching { it.close() } }
        g.model.close()
    }

    private fun cacheAllowed(spec: GraphSpec, graphFile: File, cacheDir: File): Boolean {
        if (cacheRoles.none { spec.role.startsWith(it) }) return false
        return roomFor(spec.role, cacheFile(cacheDir, spec), graphFile)
    }

    private fun roomFor(key: String, cache: File, graphFile: File): Boolean {
        if (cache.isFile) {
            cacheDecisions[key] = "on: cache file present (${cache.length()} B)"
            return true
        }
        val free = cache.absoluteFile.parentFile?.usableSpace ?: 0L
        val need = graphFile.length() * CACHE_FACTOR + CACHE_MARGIN
        val ok = free >= need
        cacheDecisions[key] = "${if (ok) "on" else "off"}: free $free B, need $need B"
        return ok
    }

    companion object {
        fun cacheFile(cacheDir: File, spec: GraphSpec): File =
            File(cacheDir, "${cacheKey(spec)}_mldrift_program_cache.bin")

        fun cacheKey(spec: GraphSpec): String = spec.file.removeSuffix(".tflite")

        fun xnnpackCacheFile(dir: File, spec: GraphSpec): File = File(dir, spec.file + XNNPACK_CACHE_SUFFIX)

        const val XNNPACK_CACHE_SUFFIX = ".xnnpack_cache"
        const val CACHE_FACTOR = 4L
        const val CACHE_MARGIN = 4L shl 30

        fun precisionOf(name: String): CompiledModel.GpuOptions.Precision =
            when (name) {
                "fp32" -> CompiledModel.GpuOptions.Precision.FP32
                "fp16acc32" -> CompiledModel.GpuOptions.Precision.FP16_WITH_FP32_ACCUM
                "fp16" -> CompiledModel.GpuOptions.Precision.FP16
                "default" -> CompiledModel.GpuOptions.Precision.DEFAULT
                else -> CompiledModel.GpuOptions.Precision.DEFAULT
            }
    }
}
