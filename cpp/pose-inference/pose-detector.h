#pragma once

#include <memory>
#include <string>

class PoseDetector
{
  public:
    // 38 54 46
    virtual void detect_pose(const std::string &image_path) = 0;
};

void create_pose_detector(std::unique_ptr<PoseDetector> &p);
