<template>
  <div>
    <WebcamModel
      modelName="Emotion"
      :modelFilepath="modelFilepath"
      :imageSize="64"
      :imageUrls="imageUrls"
      :warmupModel="warmupModel"
      :preprocess="preprocess"
      :postprocess="postprocess"
    ></WebcamModel>
    <canvas id="temp-canvas" v-show="false" />
  </div>
</template>

<script lang="ts">
import WebcamModel from "../common/WebcamModelUI.vue";
import { Vue, Component } from "vue-property-decorator";
import { runModelUtils } from "../../utils/index";
import { EMOTION_IMAGE_URLS } from "../../data/sample-image-urls";
import { Tensor, InferenceSession } from "onnxruntime-web";

import { softmax } from "../../utils/math";

const MODEL_FILEPATH_PROD = `/onnxruntime-web-demo/emotion.onnx`;
const MODEL_FILEPATH_DEV = "/emotion.onnx";

@Component({
  components: {
    WebcamModel,
  },
})
export default class Emotion extends Vue {
  imageUrls: Array<{ text: string; value: string }>;
  modelFilepath: string;

  constructor() {
    super();
    this.imageUrls = EMOTION_IMAGE_URLS;
    this.modelFilepath =
      process.env.NODE_ENV === "production"
        ? MODEL_FILEPATH_PROD
        : MODEL_FILEPATH_DEV;
  }

  warmupModel(session: InferenceSession) {
    return runModelUtils.warmupModel(session, [1, 1, 64, 64]);
  }

  preprocess(ctx: CanvasRenderingContext2D): Tensor {
    const data = this.scale(ctx);
    const width = 64;
    const height = 64;
    // data processing
    const greyScale = [];
    for (let i = 0; i < data.length; i += 4) {
      greyScale.push(
        (data[i] * 0.299 + data[i + 1] * 0.587 + data[i + 2] * 0.114 - 127.5) /
          127.5
      );
    }
    const tensor = new Tensor("float32", new Float32Array(width * height), [
      1,
      1,
      width,
      height,
    ]);
    (tensor.data as Float32Array).set(greyScale);
    return tensor;
  }

  scale(ctx: CanvasRenderingContext2D): Uint8ClampedArray {
    const scaledImage = document.getElementById(
      "temp-canvas"
    ) as HTMLCanvasElement;
    const scaledCtx = scaledImage.getContext("2d") as CanvasRenderingContext2D;
    scaledImage.width = 64;
    scaledImage.height = 64;
    scaledCtx.drawImage(ctx.canvas, 0, 0, 64, 64);
    return scaledCtx.getImageData(0, 0, 64, 64).data;
  }

  postprocess(tensor: Tensor, inferenceTime: number) {
    const output = tensor.data;
    const emotionMap = [
      "neutral",
      "happiness",
      "surprise",
      "sadness",
      "anger",
      "disgust",
      "fear",
      "contempt",
    ];
    const myOutput = softmax(Array.prototype.slice.call(output));

    let maxInd = -1;
    let maxProb = -1;
    for (let i = 0; i < myOutput.length; i++) {
      if (maxProb < myOutput[i]) {
        maxProb = myOutput[i];
        maxInd = i;
      }
    }

    this.drawRect(
      416 / 2 - 75,
      0,
      150,
      50,
      `${emotionMap[maxInd]}\nTime: ${inferenceTime.toFixed(1)}ms`
    );
  }

  drawRect(
    x: number,
    y: number,
    w: number,
    h: number,
    text = "",
    color = "blue"
  ) {
    const rect = document.createElement("div");
    const label = document.createElement("div");
    rect.style.cssText = `top: 0px;`;
    label.style.cssText = "font-size: 24px";
    label.innerText = text;
    rect.appendChild(label);

    (document.getElementById("webcam-container") as HTMLElement).appendChild(
      rect
    );
  }
}
</script>