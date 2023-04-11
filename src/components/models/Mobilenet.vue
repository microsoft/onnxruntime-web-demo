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
import type { ClassResult } from "../../utils/imagenet";
import { MOBILENET_IMAGE_URLS } from "../../data/sample-image-urls";
import { imagenetUtils, mathUtils } from "../../utils/index";

import { defineComponent } from "vue";
// import type { PropType } from "vue";

const MODEL_FILEPATH = "..\\src\\assets\\Models\\mobilenetv2-7.onnx";

export default defineComponent({
  name: "MobileNet",
  components: {
    ImageModelUI,
  },

  setup(){
    const modelFilepath = MODEL_FILEPATH;
    const imageUrls = MOBILENET_IMAGE_URLS;  

    function preprocess(ctx: CanvasRenderingContext2D): Tensor {
      const imageData = ctx.getImageData(
        0,
        0,
        ctx.canvas.width,
        ctx.canvas.height
      );

      // R - ((R/255)-0.485)/0.229 --> R/(255*0.229) - 0.485/0.229 --> (R - 0.485*255)/(255*0.229)
      // G - ((R/255)-0.456)/0.224 --> G/(255*0.224) - 0.456/0.224 --> (G - 0.456*255)/(255*0.224)
      // B - ((R/255)-0.406)/0.225 --> B/(255*0.225) - 0.406/0.225 --> (B - 0.406*255)/(255*0.225)
      // const tensor = Tensor.fromImage(imageData,{norm:{bias:[123.675,116.28,103.53,0],mean:[0.229,0.224,0.255,0]}});
      console.log("preprocess");
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

    function getPredictedClass(res: Float32Array): ClassResult[] {
      // console.log("getPredictedClass");
      if (!res || res.length === 0) {
        const empty = [] as ClassResult[];
        for (let i = 0; i < 5; i++) {
          empty.push({ id: '0', name: "-", probability: 0, index: 0 });
        }
        return empty;
      }
      // console.log("res - " + res);
      const output = mathUtils.softmax(Array.prototype.slice.call(res));
      // console.log("output - " + output);
      return imagenetUtils.imagenetClassesTopK(output, 5);
      // console.log("ans[0] - " + ans[0], ans[1], ans[2], ans[3], ans[4]);
      // return ans;
    }

    return {
      getPredictedClass,
      preprocess,
      imageUrls,
      modelFilepath,
    }
  },

})
</script>
