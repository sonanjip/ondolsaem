package io.github.taehunkim.ondolsaem;

import android.os.ParcelFileDescriptor;

/** Category "decision" (API 3, entailImage in API 4). */
interface IDecisionService {
    /**
     * For each hypothesis, the probability (0..1) that it follows from [premise] (natural language inference:
     * entailment against contradiction). Long premises are cut to the model's window.
     */
    float[] entail(String premise, in String[] hypotheses, int priority) = 1;

    /**
     * API 4: For each hypothesis, the probability (0..1) that it follows from [image] and optional [premise].
     */
    float[] entailImage(in ParcelFileDescriptor image, String premise, in String[] hypotheses, int priority) = 2;
}
