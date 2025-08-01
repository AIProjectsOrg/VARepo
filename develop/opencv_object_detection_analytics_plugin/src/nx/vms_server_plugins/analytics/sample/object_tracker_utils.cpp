// Copyright 2018-present Network Optix, Inc. Licensed under MPL 2.0: www.mozilla.org/MPL/2.0/

#include "object_tracker_utils.h"

#include "geometry.h"
#include "config.h"

namespace sample_company {
namespace vms_server_plugins {
namespace opencv_object_detection {

using namespace cv;
using namespace cv::detail::tracking::tbm;

using namespace nx::sdk;

namespace {

/**
 * Magnifies a bounding box while keeping it centered and within image boundaries
 */
cv::Rect magnifyBoundingBox(
    const cv::Rect& bbox, 
    float scaleFactor,
    int imageWidth,
    int imageHeight)
{
    // Calculate center point
    float centerX = bbox.x + bbox.width / 2.0f;
    float centerY = bbox.y + bbox.height / 2.0f;

    // Calculate new dimensions
    int newWidth = static_cast<int>(bbox.width * scaleFactor);
    int newHeight = static_cast<int>(bbox.height * scaleFactor);

    // Calculate new top-left corner while keeping box centered
    int newX = static_cast<int>(centerX - newWidth / 2.0f);
    int newY = static_cast<int>(centerY - newHeight / 2.0f);

    // Ensure the box stays within image boundaries
    newX = std::max(0, std::min(newX, imageWidth - newWidth));
    newY = std::max(0, std::min(newY, imageHeight - newHeight));
    newWidth = std::min(newWidth, imageWidth - newX);
    newHeight = std::min(newHeight, imageHeight - newY);

    return cv::Rect(newX, newY, newWidth, newHeight);
}

/**
 * Returns a rectangle that is 'factor' times **smaller** but still centred on the
 * same point and clamped inside the image – the exact inverse of magnifyBoundingBox().
 */
static cv::Rect shrinkBoundingBox(
    const cv::Rect& bbox,
    float factor,
    int imageWidth,
    int imageHeight)
{
    return magnifyBoundingBox(bbox, 1.0f / factor, imageWidth, imageHeight);
}

} // namespace

Uuid IdMapper::get(int64_t id)
{
    const auto it = m_map.find(id);
    if (it == m_map.end())
    {
        Uuid result = UuidHelper::randomUuid();
        m_map[id] = result;
        return result;
    }
    return it->second;
}

void IdMapper::removeAllExcept(const std::set<Uuid>& idsToKeep)
{
    for (auto it = m_map.begin(); it != m_map.end(); )
    {
        if (idsToKeep.find(it->second) == idsToKeep.end())
            it = m_map.erase(it);
        else
            ++it;
    }
}

/**
 * Convert detections from the plugin format to the format of opencv::detail::tracking::tbm, preserving classLabels.
 */
TrackedObjects convertDetectionsToTrackedObjects(
    const Frame& frame,
    const DetectionList& detections,
    ClassLabelMap* inOutClassLabels)
{
    TrackedObjects result;
    
    for (const std::shared_ptr<Detection>& detection: detections)
    {
        cv::Rect cvRect = nxRectToCvRect(
            detection->boundingBox,
            frame.width,
            frame.height);

        // Magnify bounding box if the detection is a cigarette
        if (detection->classLabel == "cigarette")
        {
            cvRect = magnifyBoundingBox(
                cvRect,
                TrackerConfig::cigaretteMagnificationFactor(),
                frame.width,
                frame.height);
        }

        inOutClassLabels->insert(std::make_pair(CompositeDetectionId{
            frame.index,
            cvRect},
            detection->classLabel));

        result.push_back(TrackedObject(
            cvRect,
            detection->confidence,
            (int) frame.index,
            /*object_id*/ -1)); //< Placeholder, to be filled in ObjectTracker::process().
    }

    return result;
}

/**
 * Convert detection from tbm format to our format, restoring the classLabels.
 */
std::shared_ptr<DetectionInternal> convertTrackedObjectToDetection(
    const Frame& frame,
    const TrackedObject& trackedDetection,
    const std::string& classLabel,
    IdMapper* idMapper)
{
    
    // Use original-sized box for metadata if this is a cigarette
    cv::Rect metaRect = trackedDetection.rect;
    if (classLabel == "cigarette")
    {
        metaRect = shrinkBoundingBox(
            metaRect,
            TrackerConfig::cigaretteMagnificationFactor(),
            frame.width,
            frame.height);
    }

    auto detection = std::make_shared<Detection>(Detection{
    /*boundingBox*/ cvRectToNxRect(metaRect, frame.width, frame.height),
        classLabel,
        (float) trackedDetection.confidence,
        /*trackId*/ idMapper->get(trackedDetection.object_id)});
    return std::make_shared<DetectionInternal>(DetectionInternal{
        detection,
        trackedDetection.object_id,
    });
}

/**
 * Convert detections from opencv::detail::tracking::tbm format to the plugin format, restoring classLabels.
 */
DetectionInternalList convertTrackedObjectsToDetections(
    const Frame& frame,
    const TrackedObjects& trackedDetections,
    const ClassLabelMap& classLabels,
    IdMapper* idMapper)
{
    DetectionInternalList result;
    for (const cv::detail::tracking::tbm::TrackedObject& trackedDetection: trackedDetections)
    {
        const std::string classLabel = classLabels.at({
            frame.index,
            trackedDetection.rect});
        result.push_back(convertTrackedObjectToDetection(
            frame,
            trackedDetection,
            classLabel,
            idMapper));
    }

    return result;
}

DetectionList extractDetectionList(const DetectionInternalList& detectionsInternal)
{
    DetectionList result;
    for (const std::shared_ptr<DetectionInternal>& detection: detectionsInternal)
        result.push_back(detection->detection);
    return result;
}

} // namespace opencv_object_detection
} // namespace vms_server_plugins
} // namespace sample_company
