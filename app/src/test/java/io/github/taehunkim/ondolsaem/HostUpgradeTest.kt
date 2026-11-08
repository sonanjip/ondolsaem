package io.github.taehunkim.ondolsaem

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class HostUpgradeTest {
    @Test fun aFreshInstallGetsTheNewDefaultAndTheFirstRunScreens() {
        val s = HostUpgrade.steps(from = 0, existingInstall = false)
        assertFalse(s.keepOldEmbeddingDefault); assertFalse(s.skipOnboarding)
    }

    @Test fun anInstallFromBeforeVersionsWereStoredSkipsOnboarding() {
        val s = HostUpgrade.steps(from = 0, existingInstall = true)
        assertFalse(s.keepOldEmbeddingDefault); assertTrue(s.skipOnboarding)
    }

    @Test fun laterUpdatesChangeNothing() {
        val s = HostUpgrade.steps(from = 400, existingInstall = true)
        assertFalse(s.keepOldEmbeddingDefault); assertFalse(s.skipOnboarding)
    }
}
