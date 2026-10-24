package com.example.agri.cropdx;

import java.util.List;

/** Mirrors POST /v1/predict. Jackson ignores unknown fields, so adding fields later will not break you. */
public record PredictionResponse(
        String requestId,
        String modelVersion,
        String status,                 // "CONFIDENT" | "LOW_CONFIDENCE"
        ClassScore prediction,         // NULL when LOW_CONFIDENCE
        List<ClassScore> candidates,   // top-k, always present
        double confidenceThreshold,
        List<String> qualityWarnings,  // IMAGE_BLURRY, IMAGE_TOO_DARK, IMAGE_TOO_BRIGHT, LOW_RESOLUTION
        String message,
        String disclaimer,             // MUST be shown to the end user
        double inferenceMs) {

    public record ClassScore(String classId, String crop, String condition,
                             String displayName, boolean healthy, double confidence) {}
}
