<template>
  <WebcamModel
    modelName="Yolo"
    :hasWebGL="hasWebGL"
    :modelFilepath="modelFilepath"
    :imageSize="416"
    :imageUrls="imageUrls"
    :warmupModel="warmupModel"
    :preprocess="preprocess"
    :postprocess="postprocess"
  ></WebcamModel>
</template>

<script lang="ts">
import ndarray from "ndarray";
import ops from "ndarray-ops";
import WebcamModel from "../common/WebcamModelUI.vue";
import { Vue, Component, Prop } from "vue-property-decorator";
import { runModelUtils, yolo, yoloTransforms } from "../../utils/index";
import { YOLO_IMAGE_URLS } from "../../data/sample-image-urls";
import { Tensor, InferenceSession } from "onnxruntime-web";

const MODEL_FILEPATH_PROD = `/onnxruntime-web-demo/yolo.onnx`;
const MODEL_FILEPATH_DEV = "/yolo.onnx";

@Component({
  components: {
    WebcamModel,
  },
})
export default class Yolo extends Vue {
  @Prop(Boolean) hasWebGL!: boolean;
  imageUrls: Array<{ text: string; value: string }>;
  modelFilepath: string;

  constructor() {
    super();
    this.imageUrls = YOLO_IMAGE_URLS;
    this.modelFilepath =
      process.env.NODE_ENV === "production"
        ? MODEL_FILEPATH_PROD
        : MODEL_FILEPATH_DEV;
  }

  warmupModel(session: InferenceSession) {
    return runModelUtils.warmupModel(session, [1, 3, 416, 416]);
  }

  preprocess(ctx: CanvasRenderingContext2D): Tensor {
    const imageData = ctx.getImageData(
      0,
      0,
      ctx.canvas.width,
      ctx.canvas.height
    );
    const { data, width, height } = imageData;
    // data processing
    const dataTensor = ndarray(new Float32Array(data), [width, height, 4]);
    const dataProcessedTensor = ndarray(new Float32Array(width * height * 3), [
      1,
      3,
      width,
      height,
    ]);

    ops.assign(
      dataProcessedTensor.pick(0, 0, null, null),
      dataTensor.pick(null, null, 0)
    );
    ops.assign(
      dataProcessedTensor.pick(0, 1, null, null),
      dataTensor.pick(null, null, 1)
    );
    ops.assign(
      dataProcessedTensor.pick(0, 2, null, null),
      dataTensor.pick(null, null, 2)
    );

    const tensor = new Tensor("float32", new Float32Array(width * height * 3), [
      1,
      3,
      width,
      height,
    ]);
    (tensor.data as Float32Array).set(dataProcessedTensor.data);
    return tensor;
  }

  async postprocess(tensor: Tensor, inferenceTime: number) {
    try {
      const originalOutput = new Tensor(
        "float32",
        tensor.data as Float32Array,
        [1, 125, 13, 13]
      );
      const outputTensor = yoloTransforms.transpose(
        originalOutput,
        [0, 2, 3, 1]
      );

      // postprocessing
      const boxes = await yolo.postprocess(outputTensor, 20);
      boxes.forEach((box) => {
        const { top, left, bottom, right, classProb, className } = box;

        this.drawRect(
          left,
          top,
          right - left,
          bottom - top,
          `${className} Confidence: ${Math.round(
            classProb * 100
          )}% Time: ${inferenceTime.toFixed(1)}ms`
        );
      });
    } catch (e) {
      alert("Model is not valid!");
    }
  }

  drawRect(
    x: number,
    y: number,
    w: number,
    h: number,
    text = "",
    color = "red"
  ) {
    const webcamContainerElement = document.getElementById("webcam-container") as HTMLElement;
    // Depending on the display size, webcamContainerElement might be smaller than 416x416.
    const [ox, oy] = [(webcamContainerElement.offsetWidth - 416) / 2, (webcamContainerElement.offsetHeight - 416) / 2];
    const rect = document.createElement("div");
    rect.style.cssText = `top: ${y+oy}px; left: ${x+ox}px; width: ${w}px; height: ${h}px; border-color: ${color};`;
    const label = document.createElement("div");
    label.innerText = text;
    rect.appendChild(label);

    webcamContainerElement.appendChild(
      rect
    );
  }
}
</script>