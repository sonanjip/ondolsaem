package io.github.taehunkim.ondolsaem.engine.iris

import java.io.File
import java.nio.file.Files
import java.security.MessageDigest

/**
 * The files of one run in the app's files dir.
 */
class RunFiles(val dir: File, configName: String = "iris3b_config_dit_int8_fp16.json") {

    class Tensor(val offset: Long, val count: Int, val shape: List<Int>)

    val configFile: File = File(dir, configName).takeIf { it.exists() }
        ?: File(dir, "iris3b_config.json").takeIf { it.exists() }
        ?: File(dir, configName)

    val config: Map<String, Any?> = obj(Json.parse(configFile.readText()))
    val residency: Residency = runCatching { Residency.valueOf(config.str("residency")) }.getOrDefault(Residency.P1)

    val dumps: List<String> = (config["dumps"] as? List<*>)?.map { it as String } ?: emptyList()

    private val hostFile = File(dir, obj(config["host"]).str("file"))
    private val hostIndex: Map<String, Tensor> = obj(obj(config["host"])["tensors"]).mapValues { (_, v) ->
        val t = obj(v)
        Tensor(t.num("offset").toLong(), t.num("count").toInt(), (t["shape"] as List<*>).map { (it as Double).toInt() })
    }
    private val hostCache = HashMap<String, FloatArray>()

    /** Host tensors from the pack, each read once. */
    val host = HostTensors { name ->
        hostCache.getOrPut(name) {
            val t = requireNotNull(hostIndex[name]) { "host tensor $name is not in ${hostFile.name}" }
            HostOps.readFloats(hostFile, t.offset, t.count)
        }
    }

    fun hostShape(name: String): List<Int> = requireNotNull(hostIndex[name]) { "no host tensor $name" }.shape

    /** The [EngineConfig] the config file describes. */
    fun engineConfig(): EngineConfig {
        val g = obj(config["graphs"])
        return EngineConfig(
            residency = residency,
            textEncoder = g.list("te").map { spec(it) },
            adapter = spec(g["adapter"]),
            dual = g.list("dual").map { spec(it) },
            dualModRows = g.list("dual_mod_rows").map { range(it) },
            single = g.list("single").map { spec(it) },
            singleModRows = g.list("single_mod_rows").map { range(it) },
            pixel = spec(g["pixel"]),
            dumps = dumps.toSet(),
            prefetch = prefetchRule(),
        )
    }

    private fun prefetchRule(): PrefetchRule {
        val p = config["prefetch"] as? Map<*, *> ?: return PrefetchRule()
        val default = PrefetchRule()
        return PrefetchRule(
            factor = (p["factor"] as? Double) ?: default.factor,
            marginBytes = (p["margin_bytes"] as? Double)?.toLong() ?: default.marginBytes,
        )
    }

    /** Every graph file the config names, in pipeline order. */
    fun graphFiles(): List<String> {
        val c = engineConfig()
        return (c.textEncoder + c.adapter + c.dual + c.single + c.pixel).map { it.file }
    }

    fun embedFile(): File = File(dir, config.str("embed"))

    /**
     * Writes the tokenizer files into `tok/`.
     */
    fun materializeTokenizer(): Map<String, String> {
        val tok = File(dir, "tok")
        tok.mkdirs()
        val files = obj(obj(Json.parse(File(dir, config.str("tokenizer")).readText()))["files"])
        val sums = LinkedHashMap<String, String>()
        for ((name, text) in files) {
            val bytes = (text as String).toByteArray(Charsets.UTF_8)
            File(tok, name).writeBytes(bytes)
            sums[name] = sha256(bytes)
        }
        runCatching {
            val link = File(tok, PromptTokens.EMBEDDING_FILE).toPath()
            Files.deleteIfExists(link)
            Files.createSymbolicLink(link, embedFile().absoluteFile.toPath())
        }
        return sums
    }

    private fun spec(any: Any?): GraphSpec {
        val m = obj(any)
        val accel = (m["accel"] as? String) ?: "gpu"
        val file = m.str("file")
        return GraphSpec(
            role = m.str("role"),
            file = file,
            precision = m.str("precision"),
            signature = m.str("signature"),
            inputs = m.list("inputs").map { it as String },
            outputs = m.list("outputs").map {
                val pair = it as List<*>
                (pair[0] as String) to (pair[1] as Double).toInt()
            },
            accel = accel,
            threads = ((m["threads"] ?: config["cpu_threads"]) as? Double)?.toInt() ?: 4,
            xnnpackCache = m["xnnpack_cache"] as? Boolean ?: false,
            bytes = File(dir, file).length(),
        )
    }

    private fun range(any: Any?): IntArray = (any as List<*>).map { (it as Double).toInt() }.toIntArray()

    companion object {
        @Suppress("UNCHECKED_CAST")
        fun obj(any: Any?): Map<String, Any?> = requireNotNull(any as? Map<String, Any?>) { "expected a JSON object" }

        fun Map<String, Any?>.str(key: String): String = requireNotNull(this[key] as? String) { "missing string $key" }

        fun Map<String, Any?>.num(key: String): Double = requireNotNull(this[key] as? Double) { "missing number $key" }

        fun Map<String, Any?>.list(key: String): List<Any?> = requireNotNull(this[key] as? List<*>) { "missing list $key" }

        fun sha256(bytes: ByteArray): String =
            MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { "%02x".format(it) }
    }
}
