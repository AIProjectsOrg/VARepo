// // Copyright 2018-present Network Optix, Inc. Licensed under MPL 2.0: www.mozilla.org/MPL/2.0/

// #include "detection.h"

// namespace sample_company {
// namespace vms_server_plugins {
// namespace opencv_object_detection {

// // Class labels for the MobileNet SSD model (VOC dataset).
// const std::vector<std::string> kClasses{
//     "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
//     "chair", "cow", "dining table", "dog", "horse", "motorbike", "person", "potted plant",
//     "sheep", "sofa", "train", "tv monitor"
// };
// const std::vector<std::string> kClassesToDetect{"cat", "dog", "person"};
// const std::map<std::string, std::string> kClassesToDetectPluralCapitalized{
//     {"cat", "Cats"}, {"dog", "Dogs"}, {"person", "People"}};

// } // namespace opencv_object_detection
// } // namespace vms_server_plugins
// } // namespace sample_company


// Copyright 2018-present Network Optix, Inc. Licensed under MPL 2.0: www.mozilla.org/MPL/2.0//

#include "detection.h"

namespace sample_company {
namespace vms_server_plugins {
namespace opencv_object_detection {

// Class labels for the YOLOv8n model (COCO dataset, 80 classes)
// const std::vector<std::string> kClasses{
//     "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
//     "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
//     "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
//     "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
//     "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
//     "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed",
//     "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
//     "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
// };


const std::vector<std::string> kClasses{
    "cigarette", "person", "smoke"
};

// // Only detect these classes
// const std::vector<std::string> kClassesToDetect{
//  "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
//     "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
//     "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
//     "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
//     "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
//     "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed",
//     "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
//     "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
// };

// const std::vector<std::string> kClassesToDetect{
//     "person",
//     "cat",
//     "dog"
// };

const std::vector<std::string> kClassesToDetect{
    "cigarette",
    "person",
    "smoke"
};

// const std::map<std::string, std::string> kClassesToDetectPluralCapitalized{
//     {"person", "People"},
//     {"bicycle", "Bicycles"},
//     {"car", "Cars"},
//     {"motorcycle", "Motorcycles"},
//     {"airplane", "Airplanes"},
//     {"bus", "Buses"},
//     {"train", "Trains"},
//     {"truck", "Trucks"},
//     {"boat", "Boats"},
//     {"traffic light", "Traffic Lights"},
//     {"fire hydrant", "Fire Hydrants"},
//     {"stop sign", "Stop Signs"},
//     {"parking meter", "Parking Meters"},
//     {"bench", "Benches"},
//     {"bird", "Birds"},
//     {"cat", "Cats"},
//     {"dog", "Dogs"},
//     {"horse", "Horses"},
//     {"sheep", "Sheep"},
//     {"cow", "Cows"},
//     {"elephant", "Elephants"},
//     {"bear", "Bears"},
//     {"zebra", "Zebras"},
//     {"giraffe", "Giraffes"},
//     {"backpack", "Backpacks"},
//     {"umbrella", "Umbrellas"},
//     {"handbag", "Handbags"},
//     {"tie", "Ties"},
//     {"suitcase", "Suitcases"},
//     {"frisbee", "Frisbees"},
//     {"skis", "Skis"},
//     {"snowboard", "Snowboards"},
//     {"sports ball", "Sports Balls"},
//     {"kite", "Kites"},
//     {"baseball bat", "Baseball Bats"},
//     {"baseball glove", "Baseball Gloves"},
//     {"skateboard", "Skateboards"},
//     {"surfboard", "Surfboards"},
//     {"tennis racket", "Tennis Rackets"},
//     {"bottle", "Bottles"},
//     {"wine glass", "Wine Glasses"},
//     {"cup", "Cups"},
//     {"fork", "Forks"},
//     {"knife", "Knives"},
//     {"spoon", "Spoons"},
//     {"bowl", "Bowls"},
//     {"banana", "Bananas"},
//     {"apple", "Apples"},
//     {"sandwich", "Sandwiches"},
//     {"orange", "Oranges"},
//     {"broccoli", "Broccolis"},
//     {"carrot", "Carrots"},
//     {"hot dog", "Hot Dogs"},
//     {"pizza", "Pizzas"},
//     {"donut", "Donuts"},
//     {"cake", "Cakes"},
//     {"chair", "Chairs"},
//     {"couch", "Couches"},
//     {"potted plant", "Potted Plants"},
//     {"bed", "Beds"},
//     {"dining table", "Dining Tables"},
//     {"toilet", "Toilets"},
//     {"tv", "TVs"},
//     {"laptop", "Laptops"},
//     {"mouse", "Mice"},
//     {"remote", "Remotes"},
//     {"keyboard", "Keyboards"},
//     {"cell phone", "Cell Phones"},
//     {"microwave", "Microwaves"},
//     {"oven", "Ovens"},
//     {"toaster", "Toasters"},
//     {"sink", "Sinks"},
//     {"refrigerator", "Refrigerators"},
//     {"book", "Books"},
//     {"clock", "Clocks"},
//     {"vase", "Vases"},
//     {"scissors", "Scissors"},
//     {"teddy bear", "Teddy Bears"},
//     {"hair drier", "Hair Driers"},
//     {"toothbrush", "Toothbrushes"}
// };


// Plural and capitalized names for event captions
// const std::map<std::string, std::string> kClassesToDetectPluralCapitalized{
//     {"person", "People"},
//     {"cat", "Cats"},
//     {"dog", "Dogs"}
// };

// Plural and capitalized names for event captions
const std::map<std::string, std::string> kClassesToDetectPluralCapitalized{
    {"cigarette", "Cigarettes"},
    {"person", "People"},
    {"smoke", "Smokes"}
};

} // namespace opencv_object_detection
} // namespace vms_server_plugins
} // namespace sample_company
