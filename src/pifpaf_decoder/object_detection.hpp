///
/// Ai-app interface for object detection
///
/// \copyright 2018 NVISO SA. All rights reserved.
/// \license This project is released under the XXXXXX License.
///

#pragma once

#include "image_based.hpp"

namespace lpdnn {
namespace ai_app {

    /// Object detection AiApp
    class Object_detection : virtual public Image_based {
    public:
        struct Result {
            struct Item {
                float confidence;
                int class_index;
                Rect bounding_box;
                Landmarks landmarks;
            };

            bool success{};
            std::vector<Item> items;
        };

        /// Set minimum detectable object size
        /// @return true if success
        virtual bool set_min_size(Dim2d minSize) = 0;

        /// Set maximum detectable object size
        /// @return true if success
        virtual bool set_max_size(Dim2d maxSize) = 0;

        /// Perform inference.
        virtual Result execute(const Image& input) = 0;

        /// @return Names of classes
        virtual std::vector<std::string> classes() = 0;

        /// @return our aiapp class id
        const char* class_id() const override { return ai_class_id; }
        static constexpr char const* ai_class_id = "com_bonseyes::object_detection";
    };

} // namespace ai_app
} // namespace lpdnn
