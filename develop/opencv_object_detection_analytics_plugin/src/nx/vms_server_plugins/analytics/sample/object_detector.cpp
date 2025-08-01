// Copyright 2018-present Network Optix, Inc. Licensed under MPL 2.0: www.mozilla.org/MPL/2.0/

#include "object_detector.h"



#include <opencv2/core.hpp>

#include "exceptions.h"
#include "frame.h"


namespace sample_company {
namespace vms_server_plugins {
namespace opencv_object_detection {

using namespace std::string_literals;

using namespace cv;
using namespace cv::dnn;




ObjectDetector::ObjectDetector(std::filesystem::path modelPath, DetectorConfig config):
    m_modelPath(std::move(modelPath)),
    m_config(std::move(config))
{
}

/**
 * Load the model if it is not loaded, do nothing otherwise. In case of errors terminate the
 * plugin and throw a specialized exception.
 */
void ObjectDetector::ensureInitialized()
{
    if (isTerminated())
    {
        throw ObjectDetectorIsTerminatedError(
            "Object detector initialization error: object detector is terminated.");
    }
    if (m_netLoaded)
        return;

    try
    {
        loadModel();
    }
    catch (const cv::Exception& e)
    {
        terminate();
        throw ObjectDetectorInitializationError("Loading model: " + cvExceptionToStdString(e));
    }
    catch (const std::exception& e)
    {
        terminate();
        throw ObjectDetectorInitializationError("Loading model: Error: "s + e.what());
    }
}

bool ObjectDetector::isTerminated() const
{
    return m_terminated;
}

void ObjectDetector::terminate()
{
    m_terminated = true;
}

DetectionList ObjectDetector::run(const Frame& frame)
{
    if (isTerminated())
        throw ObjectDetectorIsTerminatedError("Detection error: object detector is terminated.");

    try
    {
        return runImpl(frame);
    }
    catch (const cv::Exception& e)
    {
        terminate();
        throw ObjectDetectionError(cvExceptionToStdString(e));
    }
    catch (const std::exception& e)
    {
        terminate();
        throw ObjectDetectionError("Error: "s + e.what());
    }
}

//-------------------------------------------------------------------------------------------------
// private

void ObjectDetector::loadModel()
{

    // static const auto modelFile = m_modelPath /
    //     std::filesystem::path("yolov8n.onnx");
    static const auto modelFile = m_modelPath / m_config.modelFileName;
    m_net = std::make_unique<Net>(readNetFromONNX(modelFile.string()));

    // Save the whether the net is loaded or not to prevent unnecessary load.
    m_netLoaded = !m_net->empty();

    if (!m_netLoaded)
        throw ObjectDetectorInitializationError("Loading model: network is empty.");
    
}


cv::Mat ObjectDetector::formatToSquare(const cv::Mat &source, int *pad_x, int *pad_y, float *scale)
{
    int col = source.cols;
    int row = source.rows;
    // int m_inputWidth = m_modelShape.width;
    // int m_inputHeight = m_modelShape.height;
    int m_inputWidth  = m_config.inputSize.width;
    int m_inputHeight = m_config.inputSize.height;

    *scale = std::min(m_inputWidth / (float)col, m_inputHeight / (float)row);
    int resized_w = col * *scale;
    int resized_h = row * *scale;
    *pad_x = (m_inputWidth - resized_w) / 2;
    *pad_y = (m_inputHeight - resized_h) / 2;

    cv::Mat resized;
    cv::resize(source, resized, cv::Size(resized_w, resized_h));
    cv::Mat result = cv::Mat::zeros(m_inputHeight, m_inputWidth, source.type());
    resized.copyTo(result(cv::Rect(*pad_x, *pad_y, resized_w, resized_h)));
    resized.release();
    return result;
}

DetectionList ObjectDetector::runImpl(const Frame& frame)
{
    
    //──────────────────────────────────────────────────────────────
    // 0. Abort early if the detector was shut down.
    //──────────────────────────────────────────────────────────────
    if (isTerminated())
        throw ObjectDetectorIsTerminatedError("Object detection error: object detector is terminated.");

    //──────────────────────────────────────────────────────────────
    // 1. Grab a *copy* of the frame’s Mat so we can mutate it.
    //──────────────────────────────────────────────────────────────
    cv::Mat modelInput;
    try
    {
        modelInput = frame.cvMat;   // shallow copy, no data copy
    }
    catch (const cv::Exception& e)
    {
        throw ObjectDetectionError("Copy frame Mat: " + cvExceptionToStdString(e));
    }

    // Detection thresholds.
    // float modelScoreThreshold      {0.45};
    // float modelNMSThreshold        {0.50};
    float modelScoreThreshold { m_config.scoreThreshold };
    float modelNMSThreshold   { m_config.nmsThreshold };

    // Model expects 320×320 square input.
    // cv::Size modelShape = {320, 320};
    cv::Size modelShape       { m_config.inputSize };

    //──────────────────────────────────────────────────────────────
    // 2. Letter-box (resize + pad) to square while remembering scale.
    //──────────────────────────────────────────────────────────────
    int pad_x, pad_y;
    float scale;

    try
    {
        modelInput = formatToSquare(modelInput, &pad_x, &pad_y, &scale);
    }
    catch (const cv::Exception& e)
    {
        throw ObjectDetectionError("formatToSquare: " + cvExceptionToStdString(e));
    }

    //──────────────────────────────────────────────────────────────
    // 3. Create a 4-D blob (N,C,H,W) in YOLOv8’s expected format.
    //──────────────────────────────────────────────────────────────
    cv::Mat blob;

    try
    {
        blob = cv::dnn::blobFromImage(
            modelInput,       // BGR image
            1.0 / 255.0,      // scale to [0,1]
            modelShape,       // 320×320
            cv::Scalar(),     // no mean subtraction
            /*swapRB =*/ true,
            /*crop   =*/ false
        );
    }
    catch (const cv::Exception& e)
    {
        throw ObjectDetectionError("blobFromImage: " + cvExceptionToStdString(e));
    }
        
    //──────────────────────────────────────────────────────────────
    // 4. Run the network: setInput → forward().
    //──────────────────────────────────────────────────────────────
    std::vector<cv::Mat> outputs;
   try
    {
        m_net->setInput(blob);
        m_net->forward(outputs, m_net->getUnconnectedOutLayersNames());
    }
    catch (const cv::Exception& e)
    {
        throw ObjectDetectionError("DNN forward: " + cvExceptionToStdString(e));
    }

    // Basic validity check.
    if (outputs.empty())
        throw ObjectDetectionError("DNN forward produced no outputs.");


    int rows = outputs[0].size[2];
    int dimensions = outputs[0].size[1];

    outputs[0] = outputs[0].reshape(1, dimensions);
    cv::transpose(outputs[0], outputs[0]);

    float *data = (float *)outputs[0].data;

    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

      for (int i = 0; i < rows; ++i)
    {
            float *classes_scores = data+4;

            cv::Mat scores(1, kClasses.size(), CV_32FC1, classes_scores);
            cv::Point class_id;
            double maxClassScore;

            minMaxLoc(scores, 0, &maxClassScore, 0, &class_id);

            if (maxClassScore > modelScoreThreshold)
            {
                confidences.push_back(maxClassScore);
                class_ids.push_back(class_id.x);

                float x = data[0];
                float y = data[1];
                float w = data[2];
                float h = data[3];

                int left = int((x - 0.5 * w - pad_x) / scale);
                int top = int((y - 0.5 * h - pad_y) / scale);

                int width = int(w / scale);
                int height = int(h / scale);

                boxes.push_back(cv::Rect(left, top, width, height));


            }

            data += dimensions;
    }

    std::vector<int> nms_result;
    cv::dnn::NMSBoxes(boxes, confidences, modelScoreThreshold, modelNMSThreshold, nms_result);

    //  ── Build result list ─────────────────────────────────────────────────────────
    DetectionList result;

    const float invW = 1.0f / frame.cvMat.cols;
    const float invH = 1.0f / frame.cvMat.rows;


    for (unsigned long i = 0; i < nms_result.size(); ++i)
    {
        int idx = nms_result[i];
        
        auto& r = boxes[idx];
        result.push_back(std::make_shared<Detection>(Detection{
            nx::sdk::analytics::Rect(
                    r.x      * invW,
                    r.y      * invH,
                    r.width  * invW,
                    r.height * invH
            ),
            kClasses[class_ids[idx]],
            confidences[idx],
            nx::sdk::UuidHelper::randomUuid()
        }));
    }

    return result;
}





} // namespace opencv_object_detection
} // namespace vms_server_plugins
} // namespace sample_company
