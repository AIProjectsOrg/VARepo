// Copyright 2018-present Network Optix, Inc. Licensed under MPL 2.0: www.mozilla.org/MPL/2.0/

#pragma once

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/dnn.hpp>

#include <nx/sdk/helpers/uuid_helper.h>
#include <nx/sdk/uuid.h>

#include "detection.h"
#include "frame.h"
#include "config.h"

namespace sample_company {
namespace vms_server_plugins {
namespace opencv_object_detection {

// struct DetectorConfig
// {
//     std::string   modelFileName  = "smoking-yolo 640.onnx";
//     cv::Size      inputSize      = {640, 640};
//     float         scoreThreshold = 0.1f;
//     float         nmsThreshold   = 0.50f;
// };

struct DetectorConfig
{
    std::string   modelFileName  = TrackerConfig::detectorModelFile();
    cv::Size      inputSize      = TrackerConfig::detectorInputSize();
    float         scoreThreshold = TrackerConfig::detectorScoreThreshold();
    float         nmsThreshold   = TrackerConfig::detectorNmsThreshold();
};


class ObjectDetector
{
public:
    explicit ObjectDetector(std::filesystem::path modelPath,  DetectorConfig config = {});

    void ensureInitialized();
    bool isTerminated() const;
    void terminate();
    DetectionList run(const Frame& frame);

private:
    void loadModel();
    DetectionList runImpl(const Frame& frame);

    cv::Mat formatToSquare(
      const cv::Mat& source,
       int* pad_x,
       int* pad_y,
    float* scale
   );

    

private:
    bool m_netLoaded = false;
    bool m_terminated = false;
    const std::filesystem::path m_modelPath;
    std::unique_ptr<cv::dnn::Net> m_net;
    DetectorConfig m_config;

};

} // namespace opencv_object_detection
} // namespace vms_server_plugins
} // namespace sample_company
