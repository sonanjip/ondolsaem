package io.github.taehunkim.ondolsaem.ui

import android.Manifest
import android.content.pm.PackageManager
import android.os.Build
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.Checkbox
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import io.github.taehunkim.ondolsaem.R
import io.github.taehunkim.ondolsaem.catalog.ModelCatalog
import io.github.taehunkim.ondolsaem.catalog.ModelCategories
import io.github.taehunkim.ondolsaem.catalog.ModelSource
import io.github.taehunkim.ondolsaem.download.ModelState
import io.github.taehunkim.ondolsaem.sdk.OndolSaem.Categories

/**
 * First-run screens: what OndolSaem is, the apps that already use it, recommended models to download, and the
 * notification permission. Shown until [onDone]; installs from before 0.4.0 skip it (HostUpgrade).
 */
@Composable
fun OnboardingScreen(vm: HostViewModel, onDone: () -> Unit) {
    var step by rememberSaveable { mutableIntStateOf(0) }
    val last = 3
    Column(Modifier.fillMaxSize().safeDrawingPadding().padding(24.dp).testTag("onboarding")) {
        Text(stringResource(R.string.onb_step, step + 1, last + 1), style = MaterialTheme.typography.labelMedium)
        Spacer(Modifier.padding(4.dp))
        Column(Modifier.weight(1f).verticalScroll(rememberScrollState()), verticalArrangement = Arrangement.spacedBy(12.dp)) {
            when (step) {
                0 -> Intro()
                1 -> Apps(vm)
                2 -> Models(vm)
                else -> NotificationsStep()
            }
        }
        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween, verticalAlignment = Alignment.CenterVertically) {
            TextButton(onClick = { step-- }, enabled = step > 0) { Text(stringResource(R.string.onb_back)) }
            if (step < last) Button(onClick = { step++ }) { Text(stringResource(R.string.onb_next)) }
            else Button(onClick = onDone, modifier = Modifier.testTag("onb_finish")) { Text(stringResource(R.string.onb_finish)) }
        }
    }
}

@Composable
private fun Intro() {
    Text(stringResource(R.string.onb_intro_title), style = MaterialTheme.typography.headlineSmall)
    Text(stringResource(R.string.onb_intro_body), style = MaterialTheme.typography.bodyMedium)
    listOf(R.string.onb_intro_point_shared, R.string.onb_intro_point_local, R.string.onb_intro_point_apps).forEach {
        Text("• " + stringResource(it), style = MaterialTheme.typography.bodyMedium)
    }
}

@Composable
private fun Apps(vm: HostViewModel) {
    val clients by vm.clients.collectAsState()
    Text(stringResource(R.string.onb_apps_title), style = MaterialTheme.typography.headlineSmall)
    if (clients.isEmpty()) {
        Text(stringResource(R.string.onb_apps_none), style = MaterialTheme.typography.bodyMedium)
    } else {
        Text(stringResource(R.string.onb_apps_body), style = MaterialTheme.typography.bodyMedium)
        clients.toSortedMap().forEach { (pkg, cats) ->
            val names = cats.mapNotNull { ModelCategories.byId(it) }.map { stringResource(it.titleRes) }
            Column {
                Text(vm.label(pkg), style = MaterialTheme.typography.titleSmall)
                Text(names.joinToString().ifEmpty { pkg }, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
        }
    }
}

/** Categories whose default model is ticked: what the connected apps asked for, else everything but chat (large). */
internal fun recommendedCategories(clients: Map<String, Set<String>>): Set<String> {
    val asked = clients.values.flatten().toSet()
    return asked.ifEmpty { setOf(Categories.EMBEDDING, Categories.OCR, Categories.ASR, Categories.FACE) }
}

@Composable
private fun Models(vm: HostViewModel) {
    val clients by vm.clients.collectAsState()
    val states by vm.modelStates.collectAsState()
    // the default model of every category that has one to download
    val defaults = ModelCategories.all.mapNotNull { cat -> ModelCatalog.byId(cat.defaultModelId)?.let { cat to it } }
    var chosen by remember(clients) { mutableStateOf(recommendedCategories(clients)) }
    var askMetered by remember { mutableStateOf(false) }
    val ctx = LocalContext.current

    Text(stringResource(R.string.onb_models_title), style = MaterialTheme.typography.headlineSmall)
    Text(stringResource(R.string.onb_models_body), style = MaterialTheme.typography.bodyMedium)
    defaults.forEach { (cat, spec) ->
        val st = states[spec.id]
        val bundled = spec.source is ModelSource.Bundled
        Row(verticalAlignment = Alignment.CenterVertically) {
            Checkbox(
                checked = bundled || cat.id in chosen,
                enabled = !bundled && st !is ModelState.Installed,
                onCheckedChange = { on -> chosen = if (on) chosen + cat.id else chosen - cat.id },
                modifier = Modifier.testTag("onb_cat_${cat.id}"),
            )
            Column(Modifier.weight(1f)) {
                Text(stringResource(cat.titleRes), style = MaterialTheme.typography.titleSmall)
                Text(
                    spec.displayName + " · " + when {
                        bundled -> stringResource(R.string.onb_models_builtin)
                        st is ModelState.Installed -> stringResource(R.string.onb_models_installed)
                        st is ModelState.Downloading || st is ModelState.WaitingForWifi || st is ModelState.Verifying || st is ModelState.Converting -> stringResource(R.string.onb_models_downloading)
                        else -> spec.sizeBytes?.let(::formatBytes) ?: "?"
                    },
                    style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
        }
    }
    val toGet = defaults.filter { (cat, spec) ->
        cat.id in chosen && spec.source !is ModelSource.Bundled && spec.source !is ModelSource.AiCore &&
            states[spec.id] !is ModelState.Installed &&
            states[spec.id] !is ModelState.Downloading && states[spec.id] !is ModelState.WaitingForWifi && states[spec.id] !is ModelState.Verifying &&
                states[spec.id] !is ModelState.Converting
    }.map { it.second }
    val total = toGet.sumOf { it.sizeBytes ?: 0L }
    fun start(wifiOnly: Boolean) { toGet.forEach { vm.download(it.id, wifiOnly) } }
    if (toGet.isNotEmpty()) OutlinedButton(
        onClick = { if (vm.isMetered()) askMetered = true else start(false) },
        modifier = Modifier.testTag("onb_download"),
    ) { Text(stringResource(R.string.onb_models_download, formatBytes(total))) }
    Text(stringResource(R.string.onb_models_later), style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
    if (askMetered) {
        MeteredDialog(
            ctx.resources.getQuantityString(R.plurals.onb_models_count, toGet.size, toGet.size), formatBytes(total),
            onDownload = { wifiOnly -> start(wifiOnly); askMetered = false },
            onCancel = { askMetered = false },
        )
    }
}

@Composable
private fun NotificationsStep() {
    val ctx = LocalContext.current
    fun granted() = Build.VERSION.SDK_INT < 33 ||
        ContextCompat.checkSelfPermission(ctx, Manifest.permission.POST_NOTIFICATIONS) == PackageManager.PERMISSION_GRANTED
    var allowed by remember { mutableStateOf(granted()) }
    val launcher = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { allowed = granted() }
    Text(stringResource(R.string.onb_notif_title), style = MaterialTheme.typography.headlineSmall)
    Text(stringResource(R.string.onb_notif_body), style = MaterialTheme.typography.bodyMedium)
    if (allowed) Text(stringResource(R.string.onb_notif_allowed), style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.primary)
    else OutlinedButton(onClick = { if (Build.VERSION.SDK_INT >= 33) launcher.launch(Manifest.permission.POST_NOTIFICATIONS) }) {
        Text(stringResource(R.string.onb_notif_allow))
    }
}
