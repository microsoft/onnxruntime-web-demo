<template>
  <div>
    <WebcamModel
      modelName="Emotion"
      :hasWebGL="hasWebGL"
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

import { defineComponent } from "vue";

import { runModelUtils } from "../../utils/index";
import { EMOTION_IMAGE_URLS } from "../../data/sample-image-urls";
import { Tensor, InferenceSession } from "onnxruntime-web";

import { softmax } from "../../utils/math";

const MODEL_FILEPATH =
  "..\\src\\assets\\Models\\emotion.onnx";

// export default class WebcamModelUI extends Vue {
export default defineComponent({
  
  name: "EmotionRecognition",

  components: {
    WebcamModel,
  },

  props : {
    hasWebGL: {
      type: Boolean,
      required: true,
    },
  },

  setup(){
    let imageUrls: Array<{ text: string; value: string }> = EMOTION_IMAGE_URLS;
    let modelFilepath: string = MODEL_FILEPATH;

    async function warmupModel(session: InferenceSession): Promise<void> {
      return runModelUtils.warmupModel(session, [1, 1, 64, 64]);
    }

    function preprocess(ctx: CanvasRenderingContext2D): Tensor {
      const data = scale(ctx);
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

    function scale(ctx: CanvasRenderingContext2D): Uint8ClampedArray {
      const scaledImage = document.getElementById(
        "temp-canvas"
      ) as HTMLCanvasElement;
      const scaledCtx = scaledImage.getContext("2d") as CanvasRenderingContext2D;
      scaledImage.width = 64;
      scaledImage.height = 64;
      scaledCtx.drawImage(ctx.canvas, 0, 0, 64, 64);
      return scaledCtx.getImageData(0, 0, 64, 64).data;
    }

    function postprocess(tensor: Tensor, inferenceTime: number) {
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

      drawRect(
        416 / 2 - 75,
        0,
        150,
        50,
        `${emotionMap[maxInd]}\nTime: ${inferenceTime.toFixed(1)}ms`
      );
    }

    function drawRect(
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

    return {
      imageUrls,
      modelFilepath,
      warmupModel,
      preprocess,
      postprocess,
    }
  },
});
</script>