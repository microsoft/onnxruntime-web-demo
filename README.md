# ONNX Runtime Web Demo

ONNX Runtime Web demo is an interactive demo portal showing real use cases running [ONNX Runtime Web](https://github.com/microsoft/onnxruntime/tree/master/js/web#readme) in VueJS. It currently supports four examples for you to quickly experience the power of ONNX Runtime Web.

The demo is available here [ONNX Runtime Web demo website](https://microsoft.github.io/onnxruntime-web-demo/).

_NOTE: Currently, the supported platforms are Edge/Chrome/Firefox/Electron/Node.js (support for other platforms is coming soon)._

## Use Cases

The demo provides four scenarios based on four different ONNX pre-trained deep learning models.

### 1. SqueezeNet

[SqueezeNet](https://github.com/onnx/models/tree/master/models/image_classification/squeezenet) is a light-weight convolutional network for image classification. In the demo, you can select or upload an image and see which category it's from in miliseconds.

### 2. ResNet-50

[ResNet-50](https://github.com/onnx/models/tree/master/models/image_classification/resnet) is a highly-accurate deep convolutional network for image classification. It is trained on 1000 pre-defined classes. Similar to the SqueezeNet demo, you can select or upload an image and see which category it's from.

### 3. FER+ Emotion Recognition

[Emotion Ferplus](https://github.com/onnx/models/tree/master/emotion_ferplus)
is a deep convolutional neural network for emotion recognition in faces. In the demo, you can choose to either select an image with any human face or to start a webcam and see what emotion it's showing.

### 4. Yolo

[Yolo](https://github.com/onnx/models/tree/master/tiny_yolov2) is a real-time neural network for object detection. It can detect 20 different objects such as person, potted plant and chair. In the demo, you can choose to either select an image or start a webcam to see what objects are in it.

### 5. MNIST

[MNIST](https://github.com/onnx/models/tree/master/mnist) is a convolutional neural network that predicts handwritten digits. In the demo, you can draw any number on the canvas and the model will tell you what number it is!

## Run ONNX Runtime Web Demo

### Install Dependencies

```
npm install
```

### Serve the demo

**Serve the demo in localhost**

```
npm run serve
```

This will start a dev server and run ONNX Runtime Web demo on your localhost.

### Deploy the demo

```
npm run build
```

This will pack the source files into `/docs` folder and be ready for deployment.

**- Electron support**

ONNX Runtime Web demo can also serve as a Windows desktop app using [Electron](https://electronjs.org/).

First create a developer build of the app by running

```
npm run build -- --mode developer
```

Then run

```
npm run electron-packager
```

This will create a new `/ONNXRuntimeWeb-demo-win32-x64` folder. Run `/ONNXRuntimeWeb-demo-win32-x64/ONNXRuntimeWeb-demo.exe` to enjoy Electron desktop app.

## Credits

This demo is adapted from [keras.js demo](https://github.com/transcranial/keras-js). Modifications have been made to the UI and the backend uses `ONNX Runtime Web`.

## Contributing

This project welcomes contributions and suggestions. Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.microsoft.com.

When you submit a pull request, a CLA-bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., label, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.
