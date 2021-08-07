<template>
  <div>
    <DrawingModelUI
      :modelFilepath="modelFilepath"
      :preprocess="preprocess"
      :postprocess="postprocess"
      :getPredictedClass="getPredictedClass"
    ></DrawingModelUI>
    <canvas
      id="input-canvas-scaled"
      width="28"
      height="28"
      style="display: none"
    ></canvas>
    <canvas id="input-canvas-centercrop" style="display: none"></canvas>
  </div>
</template>

<script lang='ts'>
import DrawingModelUI from "../common/DrawingModelUI.vue";
import { Vue, Component } from "vue-property-decorator";
import { Tensor } from "onnxruntime-web";
import { mathUtils } from "../../utils";

const MODEL_FILEPATH_PROD = `/onnxruntime-web-demo/mnist.onnx`;
const MODEL_FILEPATH_DEV = "/mnist.onnx";

@Component({
  components: {
    DrawingModelUI,
  },
})
export default class MNIST extends Vue {
  modelFilepath: string;
  constructor() {
    super();
    this.modelFilepath =
      process.env.NODE_ENV === "production"
        ? MODEL_FILEPATH_PROD
        : MODEL_FILEPATH_DEV;
  }

  preprocess(ctx: CanvasRenderingContext2D): Tensor {
    // center crop
    const imageDataCenterCrop = mathUtils.centerCrop(
      ctx.getImageData(0, 0, ctx.canvas.width, ctx.canvas.height)
    );
    const ctxCenterCrop = (
      document.getElementById("input-canvas-centercrop") as HTMLCanvasElement
    ).getContext("2d") as CanvasRenderingContext2D;
    ctxCenterCrop.canvas.width = imageDataCenterCrop.width;
    ctxCenterCrop.canvas.height = imageDataCenterCrop.height;
    ctxCenterCrop.putImageData(imageDataCenterCrop, 0, 0);
    // scaled to 28 x 28
    const ctxScaled = (
      document.getElementById("input-canvas-scaled") as HTMLCanvasElement
    ).getContext("2d") as CanvasRenderingContext2D;
    ctxScaled.save();
    ctxScaled.scale(
      28 / ctxCenterCrop.canvas.width,
      28 / ctxCenterCrop.canvas.height
    );
    ctxScaled.clearRect(
      0,
      0,
      ctxCenterCrop.canvas.width,
      ctxCenterCrop.canvas.height
    );
    ctxScaled.drawImage(
      document.getElementById("input-canvas-centercrop") as HTMLCanvasElement,
      0,
      0
    );
    const imageDataScaled = ctxScaled.getImageData(
      0,
      0,
      ctxScaled.canvas.width,
      ctxScaled.canvas.height
    );
    ctxScaled.restore();
    // process image data for model input
    const { data } = imageDataScaled;
    const input = new Float32Array(784);
    for (let i = 0, len = data.length; i < len; i += 4) {
      input[i / 4] = data[i + 3] / 255;
    }

    const tensor = new Tensor("float32", input, [1, 1, 28, 28]);
    return tensor;
  }

  postprocess(rawOutput: Tensor): Float32Array {
    return mathUtils.softmax(Array.prototype.slice.call(rawOutput.data));
  }

  getPredictedClass(output: Float32Array) {
    if (output.reduce((a, b) => a + b, 0) === 0) {
      return -1;
    }
    return output.reduce(
      (argmax, n, i) => (n > output[argmax] ? i : argmax),
      0
    );
  }
}
</script>