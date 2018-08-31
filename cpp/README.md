# Openpose Inference Application in C++

## Requirements

- [bazel](http://github.com/bazelbuild/bazel), a build tool used by tensorflow.
  - install on ubuntu

    ```bash
    echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
    curl -s https://bazel.build/bazel-release.pub.gpg | sudo apt-key add -
    sudo apt update
    sudo apt install -y bazel
    ```

- A clone of tensorflow source repository: C++ tensorflow application requires in tree build.
