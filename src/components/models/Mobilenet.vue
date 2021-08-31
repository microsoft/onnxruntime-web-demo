<template>
  <ImageModelUI
    :modelFilepath="modelFilepath"
    :imageSize="224"
    :imageUrls="imageUrls"
    :preprocess="preprocess"
    :getPredictedClass="getPredictedClass"
  ></ImageModelUI>
</template>

<script lang="ts">
import ndarray from "ndarray";
import ops from "ndarray-ops";
import ImageModelUI from "../common/ImageModelUI.vue";
import { Tensor } from "onnxruntime-web";
import { Vue, Component } from "vue-property-decorator";
import { MOBILENET_IMAGE_URLS } from "../../data/sample-image-urls";
import { imagenetUtils, mathUtils } from "../../utils/index";

const MODEL_FILEPATH_PROD = `/onnxruntime-web-demo/mobilenetv2-7.onnx`;
const MODEL_FILEPATH_DEV = "/mobilenetv2-7.onnx";

@Component({
  components: {
    ImageModelUI,
  },
})
export default class MobileNet extends Vue {
  modelFilepath: string;
  imageUrls: Array<{ text: string; value: string }>;

  constructor() {
    super();
    this.modelFilepath =
      process.env.NODE_ENV === "production"
        ? MODEL_FILEPATH_PROD
        : MODEL_FILEPATH_DEV;
    this.imageUrls = MOBILENET_IMAGE_URLS;
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

    ops.divseq(dataProcessedTensor, 255);
    ops.subseq(dataProcessedTensor.pick(0, 0, null, null), 0.485);
    ops.subseq(dataProcessedTensor.pick(0, 1, null, null), 0.456);
    ops.subseq(dataProcessedTensor.pick(0, 2, null, null), 0.406);

    ops.divseq(dataProcessedTensor.pick(0, 0, null, null), 0.229);
    ops.divseq(dataProcessedTensor.pick(0, 1, null, null), 0.224);
    ops.divseq(dataProcessedTensor.pick(0, 2, null, null), 0.225);

    const tensor = new Tensor("float32", new Float32Array(width * height * 3), [
      1,
      3,
      width,
      height,
    ]);
    (tensor.data as Float32Array).set(dataProcessedTensor.data);
    return tensor;
  }

  getPredictedClass(res: Float32Array): {} {
    if (!res || res.length === 0) {
      const empty = [];
      for (let i = 0; i < 5; i++) {
        empty.push({ name: "-", probability: 0, index: 0 });
      }
      return empty;
    }
    const output = mathUtils.softmax(Array.prototype.slice.call(res));
    return imagenetUtils.imagenetClassesTopK(output, 5);
  }
}
</script>
