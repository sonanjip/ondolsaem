package io.github.taehunkim.ondolsaem.runtime

import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.test.advanceTimeBy
import kotlinx.coroutines.test.runCurrent
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

@OptIn(ExperimentalCoroutinesApi::class)
class LeasedResourceTest {
    private class Res : AutoCloseable {
        var closed = false
        override fun close() { closed = true }
    }

    @Test fun idlePolicyUnloadsAfterTwoMinutes() = runTest {
        val res = LeasedResource<Res>("t", backgroundScope, clock = { testScheduler.currentTime }) { Residency.IDLE_2M }
        res.acquire("k") { Res() }!!.close(); runCurrent()
        advanceTimeBy(119_000); runCurrent()
        assertTrue(res.isLoaded)
        advanceTimeBy(2_000); runCurrent()
        assertFalse(res.isLoaded)
    }

    @Test fun immediatePolicyUnloadsOnRelease() = runTest {
        val res = LeasedResource<Res>("t", backgroundScope, clock = { testScheduler.currentTime }) { Residency.IMMEDIATE }
        val r = res.acquire("k") { Res() }!!
        assertTrue(res.isLoaded)
        r.close(); runCurrent()
        assertFalse(res.isLoaded)
    }

    @Test fun keepPolicyNeverUnloads() = runTest {
        val res = LeasedResource<Res>("t", backgroundScope, clock = { testScheduler.currentTime }) { Residency.KEEP }
        res.acquire("k") { Res() }!!.close(); runCurrent()
        advanceTimeBy(24 * 3600_000L); runCurrent()
        assertTrue(res.isLoaded)
    }

    @Test fun neverUnloadedWhileLeased() = runTest {
        val res = LeasedResource<Res>("t", backgroundScope, clock = { testScheduler.currentTime }) { Residency.IMMEDIATE }
        val a = res.acquire("k") { Res() }!!
        val b = res.acquire("k") { error("must reuse") }!!
        a.close(); runCurrent()
        assertTrue("still leased by b", res.isLoaded)
        b.close(); runCurrent()
        assertFalse(res.isLoaded)
    }

    @Test fun reacquireCancelsPendingUnload() = runTest {
        var loads = 0
        val res = LeasedResource<Res>("t", backgroundScope, clock = { testScheduler.currentTime }) { Residency.IDLE_2M }
        res.acquire("k") { loads++; Res() }!!.close(); runCurrent()
        advanceTimeBy(100_000); runCurrent()
        val again = res.acquire("k") { loads++; Res() }!!
        advanceTimeBy(60_000); runCurrent() // past the original deadline
        assertTrue(res.isLoaded)
        assertEquals("reused, not reloaded", 1, loads)
        again.close()
    }

    @Test fun differentKeyReplacesAndClosesOld() = runTest {
        val res = LeasedResource<Res>("t", backgroundScope, clock = { testScheduler.currentTime }) { Residency.KEEP }
        val first = res.acquire("a") { Res() }!!
        val firstRes = first.value
        first.close(); runCurrent()
        res.acquire("b") { Res() }!!
        assertTrue(firstRes.closed)
    }

    @Test fun trimOnlyWhenIdle() = runTest {
        val res = LeasedResource<Res>("t", backgroundScope, clock = { testScheduler.currentTime }) { Residency.KEEP }
        val l = res.acquire("k") { Res() }!!
        res.trim(); assertTrue(res.isLoaded)
        l.close(); runCurrent()
        res.trim(); assertFalse(res.isLoaded)
    }

    @Test fun nullLoadYieldsNullLease() = runTest {
        val res = LeasedResource<Res>("t", backgroundScope, clock = { testScheduler.currentTime }) { Residency.KEEP }
        assertEquals(null, res.acquire("k") { null })
    }

    @Test fun doubleCloseReleasesOnce() = runTest {
        val res = LeasedResource<Res>("t", backgroundScope, clock = { testScheduler.currentTime }) { Residency.IMMEDIATE }
        val a = res.acquire("k") { Res() }!!
        val b = res.acquire("k") { Res() }!!
        a.close(); a.close(); runCurrent()
        assertTrue("b still holds it", res.isLoaded)
        b.close(); runCurrent()
        assertFalse(res.isLoaded)
    }

    @Test fun tracksHoldersPerOwner() = runTest {
        val res = LeasedResource<Res>("t", backgroundScope, clock = { testScheduler.currentTime }) { Residency.KEEP }
        val a = res.acquire("k", "app.a") { Res() }!!
        val b = res.acquire("k", "app.b") { Res() }!!
        res.acquire("k", "app.a") { Res() }!!.close(); runCurrent()
        assertEquals(mapOf("app.a" to 1, "app.b" to 1), res.state.value!!.holders)
        a.close(); b.close(); runCurrent()
        assertEquals(emptyMap<String, Int>(), res.state.value!!.holders)
    }

    @Test fun reportsUnloadDeadline() = runTest {
        val res = LeasedResource<Res>("t", backgroundScope, clock = { testScheduler.currentTime }) { Residency.IDLE_1M }
        res.acquire("k") { Res() }!!.close(); runCurrent()
        assertEquals(testScheduler.currentTime + 60_000, res.state.value!!.unloadAt)
        advanceTimeBy(61_000); runCurrent()
        assertEquals(null, res.state.value)
    }

    @Test fun policyChangeAppliesToIdleResource() = runTest {
        var policy = Residency.KEEP
        val res = LeasedResource<Res>("t", backgroundScope, clock = { testScheduler.currentTime }) { policy }
        res.acquire("k") { Res() }!!.close(); runCurrent()
        assertTrue(res.isLoaded)
        policy = Residency.IMMEDIATE
        res.reapplyPolicy()
        assertFalse(res.isLoaded)
    }

    @Test fun measuresMemoryGrowthWhileLoading() = runTest {
        var pss = 1_000L
        val res = LeasedResource<Res>("t", backgroundScope, measure = { pss }, clock = { 0L }) { Residency.KEEP }
        res.acquire("k") { pss += 500; Res() }!!
        assertEquals(500L, res.state.value!!.loadedKb)
    }

    @Test fun trimWaitsForPendingAsyncCloseAndUnloads() = runTest {
        val res = LeasedResource<Res>("t", backgroundScope, clock = { testScheduler.currentTime }) { Residency.IDLE_2M }
        val r = Res()
        val lease = res.acquire("k") { r }!!
        lease.close()
        // No runCurrent() before trim(): trim must still observe the release once close() has been called.
        assertTrue(res.trim())
        assertFalse(res.isLoaded)
        assertTrue(r.closed)
    }
}
