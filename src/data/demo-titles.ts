export const DEMO_TITLES: { [key: string]: string } = {
  mobilenet: "MobileNet, trained on ImageNet",
  squeezenet: "SqueezeNet, trained on ImageNet",
  emotion: "FER+ Emotion, real-time emotion detection",
  YoLo: "Yolo, real-time object detection",
  MNIST: "MNIST, handwritten digit prediction",
};

export const DEMO_DESCRIPTIONS: { [key: string]: string } = {
  mobilenet:
    "MobileNet models perform image classification - they are also very efficient in terms of speed and size and hence are ideal for embedded and mobile applications.",
  squeezenet:
    "SqueezeNet is a light-weight convolutional networks for image classification.",
  emotion:
    "FER+ Emotion, a deep convolutional neural network for emotion recognition in faces.",
  YoLo: "YOLO can detect multiple objects in an image and puts bounding boxes around these objects.",
  MNIST: "MNIST, handwritten digit prediction.",
};

export const DEMO_MODEL_LINKS: { [key: string]: string } = {
  mobilenet:
    "https://github.com/onnx/models/tree/master/vision/classification/mobilenet",
  squeezenet:
    "https://github.com/onnx/models/tree/master/vision/classification/squeezenet",
  emotion:
    "https://github.com/onnx/models/tree/master/vision/body_analysis/emotion_ferplus",
  YoLo: "https://github.com/onnx/models/tree/master/vision/object_detection_segmentation/tiny-yolov2",
  MNIST:
    "https://github.com/onnx/models/tree/master/vision/classification/mnist",
};
