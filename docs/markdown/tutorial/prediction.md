# Tutorial for Prediction Library

The prediction library of hyperpose provides 2 APIs styles:

- **Operator API**: Imperative style. (more user manipulation space)
- **Stream API**: Declarative style. (faster and simpler)

This tutorial will show you how to use them in C++ step by step. For more detailed instructions, please refer to our C++ API documents.

## End-2-end Prediction Using Stream API

In this section, we'll try to process a video via Stream API.

> Before all, please make sure you have the library successfully built(See [installation](../install/prediction.md)). 
> And we encourage you to build the tutorial examples under the folder `hyperpose/examples/user_codes`.

```bash
cd examples/user_codes
touch main.cpp

# Open your editor and do coding.

cd ../..
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=RELEASE -DBUILD_USER_CODES=ON # BUILD_USER_CODES is by default on
cmake --build .

# Execute your codes.
```

### Include the header

```c++
#include <hyperpose/hyperpose.hpp>
```

The file arrangement of include files:

- hyperpose
  - hyperpose.hpp (include all necessary headers)
  - operator (headers about Operator API)
  - stream (headers about Stream API)
  - utility

Usually, you only need to include `hyperpose/hyperpose.hpp`.

### Prepare Your Model

We support 3 types of model file:

- **Uff**: Users need to specify the input/output nodes of the network compute graph.
- **ONNX**: No input/output nodes information is required.
- **Cuda Engine Protobuf**: When importing Uff / ONNX models, TensorRT will do profiling to build the "best" runtime engine. To save the building time, we can export the model to `Cuda Engine Protobuf` format and reload it in next execution.

```c++
using namespace hyperpose;

// To use a Uff model, users needs to specify the input/output nodes.
// Here, `image` is the input node name, and `outputs/conf` and `outputs/paf` are the output feature maps. (related to the PAF algorithm)
const dnn::uff uff_model{ "../data/models/hao28-600000-256x384.uff", "image", {"outputs/conf", "outputs/paf"} };
```

### Create Input / Output Stream

We support `std::vector<cv::Mat>`, `cv::Mat`, `cv::VideoCapture` as the inputs of input stream.

We also support `cv::VideoWriter`, `NameGenerator`(a callable object which generate next name for the output image) as output streams.

```c++
// For best performance, HyperPose only allows models who have fixed input network resolution.
// What is "network resolution"? Say that the input of networks are NCHW format, the "HW" is the network resolution.
const cv::Size network_resolution{384, 256};

// * Input video.
auto capture = cv::VideoCapture("../data/media/video.avi");

// * Output video.
auto writer = cv::VideoWriter(
    "output.avi", 
    capture.get(cv::CAP_PROP_FOURCC), capture.get(cv::CAP_PROP_FPS), 
    network_resolution); // Here we use the network resolution as the output video resolution.
```

### Create DNN Engine and Post-processing Parser

```c++
// * Create TensorRT engine.
dnn::tensorrt engine(uff_model, network_resolution);

// * post-processing: Using paf.
parser::paf parser{};
```

### Create Stream Scheduler

```c++
// * Create stream
auto stream = make_stream(engine, parser);
```

### Connect the Stream

```c++
// * Connect input stream.
stream.async() << capture;

// * Connect ouput stream and wait.
stream.sync() >> writer;
```

- We provides 2 ways of stream connection.
  - `.async()`
    - Input Stream: The stream scheduler will push images asynchronously. (not blocked)
    - Output Stream: The stream scheduler will generate results asynchronously. (not blocked)
  - `.sync()`
    - Input Stream: Blocked.  This may cause deadlock if you trying to push a big number of images in a synchronous way(the buffer queue is of fixed size).
    - Output Stream: Blocked until all outputs are generated.

> We recommend you to set inputs via `async()` and generate results via `sync()`.

### Full example

Full examples are available [here](../design/design.md).

## Prediction Using Operator API

### Preparation

```c++
#include <hyperpose/hyperpose.hpp>

int main() {
    using namespace hyperpose;

    const cv::Size network_resolution{384, 256};
    const dnn::uff uff_model{ "../data/models/TinyVGG-V1-HW=256x384.uff", "image", {"outputs/conf", "outputs/paf"} };

    // * Input video.
    auto capture = cv::VideoCapture("../data/media/video.avi");

    // * Output video.
    auto writer = cv::VideoWriter(
        "output.avi", capture.get(cv::CAP_PROP_FOURCC), capture.get(cv::CAP_PROP_FPS), network_resolution);

    // * Create TensorRT engine.
    dnn::tensorrt engine(uff_model, network_resolution);

    // * post-processing: Using paf.
    parser::paf parser{};
    
    while (capture.isOpened()) {
        // ..... Applying pose estimation in one batch. 
    }
}
```

### Apply One Batch of Frames

#### Accumulate One Batch

```c++
std::vector<cv::Mat> batch;

// The .max_batch_size() of dnn::tensorrt is set in the initializer. (by default -> 8)
// initializer(model_config, input_size, max_batch_size = 8, dtype = float, factor = 1./255, flip_rgb = true)
for (int i = 0; i < engine.max_batch_size(); ++i) {
    cv::Mat mat;
    capture >> mat;
    if (mat.empty()) // If the video ends, break.
        break;
    batch.push_back(mat);
}

// Now we got a batch of images. -> batch.
```

#### Get Feature/Activation Maps

```c++
// * TensorRT Inference.
std::vector<internal_t> feature_map_packets = engine.inference(batch);
// using internal_t = std::vector<feature_map_t>
// Here, feature_map_packets = batch_size * feature_map_count(conf and paf) * feature_map.
```

#### Get Human Topology

One image may contain many humans. So the return type is `std::vector<human_t>`.

```c++
// * Paf.
std::vector<std::vector<human_t>> pose_vectors; // image_count * humans
for (auto& packet : feature_map_packets)
    pose_vectors.push_back(parser.process(packet[0]/* conf */, packet[1] /* paf */));
```

#### Visualization

```c++
// * Visualization
for (size_t i = 0; i < batch.size(); ++i) {
    cv::resize(batch[i], batch[i], network_resolution);
    for (auto&& pose : pose_vectors[i])
        draw_human(batch[i], pose); // Visualization.
    writer << batch[i];
}
```

### Full example

Full examples are available [here](../design/design.md).