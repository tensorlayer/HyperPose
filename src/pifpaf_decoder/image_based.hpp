///
/// Ai-app interface and types for image-based ai-apps
///
/// \copyright 2018 NVISO SA. All rights reserved.
/// \license This project is released under the XXXXXX License.
///

#pragma once

#include "aiapp.hpp"

namespace lpdnn {
namespace ai_app {

    /// 2-dimensional size
    struct Dim2d {
        int x;
        int y;
    };

    /// Rectangle
    struct Rect {
        Dim2d origin;
        Dim2d size;

        bool empty() const { return size.x <= 0 || size.y <= 0; }
    };

    /// Landmarks
    struct Landmark {
        Dim2d position;
        float confidence; /// Negative value if N/A
    };

    struct Landmarks {
        /// Landmark specification identifier
        std::string type;
        /// Landmark points
        std::vector<Landmark> points;
    };

    /// Image representation.
    /// The data of a RAW image consists of *y scanlines of *x pixels,
    /// with each pixel consisting of N interleaved 8-bit components; the first
    /// pixel pointed to is top-left-most in the image. There is no padding between
    /// image scanlines or between pixels, regardless of format. The number of
    /// components N is 3 for RGB images, 4 for RGBA, 1 for grayscale.
    /// Support for 8bits RGB format is MANDATORY for all image-processing AiApps.
    /// An image can be constructed from a std::vector<uint8_t>, or a std::string
    /// or raw data pointer and size. When passing rvalues vector or strings, the
    /// image will take ownership of the data, otherwise will just keep reference.
    class Image {
    protected:
        /// Contains image data if we have ownership of it
        std::vector<uint8_t> _image_content;

    public:
        /// Image format
        enum class Format {
            raw_grayscale = 1, /// 8bits grayscale
            raw_rgb8 = 3, /// 8bits RGB *MANDATORY*
            raw_rgba8 = 4, /// 8bits RGBA

            encoded = 256, /// Standard JPEG/BMP/PNG/TIFF format

            custom = 512 /// Custom format. Use attributes field for more details.
        };

        /// Don't take data ownership.
        /// img_dim parameter can be omitted in case of encoded images since
        /// this information will be extracted from the image content itself.
        Image(Format img_format, const std::vector<uint8_t>& data, Dim2d img_dim = {})
            : Image(img_format, data.data(), data.size(), img_dim)
        {
        }

        /// Take data ownership
        Image(Format img_format, std::vector<uint8_t>&& data, Dim2d img_dim = {})
            : _image_content(std::move(data))
            , format{ img_format }
            , dim(img_dim)
            , data{ _image_content.data() }
            , data_size{ _image_content.size() }
        {
        }

        /// Don't take data ownership.
        Image(Format img_format, const std::string& data, Dim2d img_dim = {})
            : Image(img_format, (uint8_t*)data.c_str(), data.size(), img_dim)
        {
        }

        /// Take data ownership
        Image(Format img_format, std::string&& data, Dim2d img_dim = {})
            : Image(img_format,
                std::vector<uint8_t>((uint8_t*)data.c_str(),
                    (uint8_t*)data.c_str() + data.size()),
                img_dim)
        {
            data.clear();
        }

        /// Don't take data ownership
        /// img_data_size is mandatory in case of encoded images.
        Image(Format img_format, const uint8_t* img_data, size_t img_data_size,
            Dim2d img_dim = {})
            : format{ img_format }
            , dim(img_dim)
            , data{ img_data }
            , data_size{ img_data_size }
        {
        }

        /// Utility factory methods
        static Image encoded(const std::vector<uint8_t>& data)
        {
            return Image(Format::encoded, data);
        }

        /// Image format
        Format format;

        /// Image dimensions (for raw images)
        Dim2d dim;

        /// Region of interest inside the image (all if empty)
        Rect roi{};

        /// Custom attributes.
        /// This is ai-app specific and allows to specify custom data formats.
        std::string attributes;

        /// Pointer to image data (no ownership of the data).
        const uint8_t* data;

        /// Size of image data. Mandatory for encoded images.
        size_t data_size;

        /// Additional optional information about the image.
        /// May be required by some aiapps.
        Landmarks landmarks;
    };

    /// Abstract image-based AiApp
    class Image_based : virtual public Aiapp {
    public:
        /// @return supported image formats (ordered by preference)
        virtual std::vector<Image::Format> image_formats() const = 0;
    };

} // namespace ai_app
} // namespace lpdnn
