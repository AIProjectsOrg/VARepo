// config.h
#pragma once
#include <opencv2/core.hpp>
#include <string>

/*  ── Central place for literals that may change ───────────────────────── */
struct TrackerConfig
{
    /* — BBox enlargement — */
    static inline float cigaretteMagnificationFactor()        { return 3.0f; }

    /* — Detector (YOLO) hyper-parameters — */
    static inline std::string detectorModelFile()             { return "smoking-yolo 640.onnx"; }
    static inline cv::Size    detectorInputSize()             { return {640, 640}; }
    static inline float       detectorScoreThreshold()        { return 0.10f; }
    static inline float       detectorNmsThreshold()          { return 0.50f; }
};
