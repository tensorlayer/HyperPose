///
/// Ai-app base interface and types
///
/// \copyright 2018 NVISO SA. All rights reserved.
/// \license This project is released under the XXXXXX License.
///

#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace lpdnn {
namespace ai_app {

/// Aiapp Blob
/// This could be improved to allow referring to existing data
/// thus avoding unneeded data-copy, for example by using shared_ptr.
struct Blob {
  /// Data dimensions. Mandatory if the blob represents a tensor.
  std::vector<int> dim;

  /// Data. Mandatory if the blob represents a tensor.
  std::vector<float> data;

  /// Optional raw representation.
  std::vector<uint8_t> raw;

  /// Optional CBOR representation when data is structured.
  std::vector<uint8_t> cbor;

  /// Optional additional information
  /// (eg, description of internal representation: "NCHW,8bits,dp3").
  std::string info;
};

/// AI-App interface
class Aiapp {
 public:
  virtual ~Aiapp() {}

  /// @return the ai-class id for this aiapp
  virtual const char* class_id() const = 0;

  /// @return the implementation id for this aiapp
  virtual const char* impl_id() const = 0;

  /// Initialization options
  /// \param cfg: configuration string, typically in JSON format.
  /// \return: true if success
  virtual bool init(const std::string& cfg) = 0;

  /// Set runtime options for the specified component
  /// \param opt: runtime options, typically in JSON format.
  /// \param name: subcomponent name
  /// \return: true if success
  virtual bool set_options(const std::string& opt,
                           const std::string& name = "") = 0;

  /// Introspection methods
  /// \{

  /// \return: names of all direct subcomponents of the specified component
  virtual std::vector<std::string> components(
      const std::string& name = "") const = 0;

  /// \return output(s) of the specified component
  virtual std::vector<Blob> output(const std::string& name = "") const = 0;

  /// \return metrics of the specified component and all its subcomponents
  virtual std::string metrics(const std::string& name = "") const = 0;

  /// set end-of-execution at the end of the specified component
  /// if name is empty any exit-point previously set is removed
  virtual bool set_exit_after(const std::string& name = "") = 0;

  /// \}
};

/// AiApp standard processing components
/// Each ai-app can contain other sub-components.
/// Each subcomponent can be identified by a pathname, for example:
///   "preprocessing.normalize"
///   "inference.net1.conv23"
struct Component {
  /// Standard component names. Their use is not mandatory but
  /// allows an ai-app to be supported by existing tools.
  static constexpr char const* preprocessing = "preprocessing";
  static constexpr char const* inference = "inference";
  static constexpr char const* postprocessing = "postprocessing";

  /// Ai-app interface parameters
  static constexpr char const* interface = "interface";

  /// Name separator in a component pathname string.
  /// Component names can't contain the separator except possibly for the leafs
  static constexpr char separator = '.';

  /// Concatenate component names in a component pathname
  static std::string join(const std::string& path, const std::string& comp) {
    return path + separator + comp;
  }
};

/// AiApp Metrics
struct Metrics {
  /// Standard metrics. All timings are in microseconds.
  static constexpr char const* init_time = "init_time";
  static constexpr char const* inference_time = "inference_time";
  static constexpr char const* inference_cpu_time = "inference_cpu_time";
};

}  // namespace ai_app
}  // namespace lpdnn
