# Performance of Prediction Library

| Method          | Backbone Size | Network Resolution | Operator API / FPS | Stream API / FPS | Other Framework / FPS |
| --------------- | ------------- | ------------------ | ------------------ | ---------------- | --------------------- |
| OpenPose COCO   | 209.3MB       | 656 x 368          | 19.78              | 27.32            | 8 (OpenPose)          |
| Tiny VGG + PAF  | 34.7 MB       | 384 x 256          | 66.62              | 124.925          | /                     |
| MobileNet + PAF | 17.9 MB       | 432 x 368          | 50.89              | 84.32            | /                     |

> **Environment**: System@Ubuntu18.04, GPU@1070Ti, CPU@i7(12 logic cores). 
>
> **Tested Video Source**: Crazy Updown Funk(resolution@640x360, frame_count@7458, source@[YouTube](https://www.youtube.com/watch?v=2DiQUX11YaY))
>
> **Availability**: All model above are available [here](https://github.com/tensorlayer/pretrained-models/tree/master/models/hyperpose). 