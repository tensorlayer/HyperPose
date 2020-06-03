#pragma once

namespace hyperpose {

namespace parser {

// The pose_proposal supports the feature map list using the following order:

// 0: predict_confidence N x 18 x 12 x 12
// 1: predict_iou        N x 18 x 12 x 12
// 2: x                  N x 18 x 12 x 12
// 3: y                  N x 18 x 12 x 12
// 4: w                  N x 18 x 12 x 12
// 5: h                  N x 18 x 12 x 12
// 6: edge_confidence    N x 17 x 9 x 9 x 12 x 12

class pose_proposal {

};

}

} // namespace hyperpose