<template>
    <div>
    <modelStatus
      v-if="modelLoading || modelInitializing"
      :modelLoading="modelLoading"
      :modelInitializing="modelInitializing"
    ></modelStatus>
    <v-container fluid style="margin-left: 25%; width: 50%; padding: 30px">
      <!-- Utility bar to select session backend configs. -->
      <v-layout
        justify-center
        align-center
        style="margin: auto; width: 100%; padding: 40px"
      >
        <div class="select-backend" style="align-self: center">
          Select Backend:
        </div>
        <v-select
          v-model="sessionBackend"
          :disabled="modelLoading || modelInitializing || sessionRunning"
          :items="backendSelectList"
          item-title="text"
          item-value="value"
          label="Switch Backend"
          :menu-props="{ maxHeight: '750' }"
          solo
          single-line
          hide-details
        ></v-select>
      </v-layout>
      <v-layout>
        <v-flex
          v-if="modelLoadingError"
          style="padding-bottom: 30px"
          class="error-message"
        >
          Error: Current backend is not supported on your machine. Try Selecting
          a different backend.
        </v-flex>
      </v-layout>
      <v-layout
        row
        wrap
        justify-center
        align-center
        class="image-panel elevation-1"
      >
        <v-flex sm6 md4>
          <div class="input-column">
            <div class="input-container">
              <div class="input-label">Draw any digit (0-9) here</div>
              <div class="canvas-container">
                <canvas
                  id="input-canvas"
                  width="300"
                  height="300"
                  @mousedown="activateDraw"
                  @mouseup="run"
                  @mouseleave="run"
                  @mousemove="draw"
                  @touchstart="activateDraw"
                  @touchmove="draw"
                ></canvas>
              </div>
            </div>
            <v-layout align-end justify-end>
              <v-btn color="primary" @click="clear" style="margin: 0px">
                <v-icon left>close</v-icon>
                Clear
              </v-btn>
            </v-layout>
          </div>
        </v-flex>
        <v-flex sm6 md4>
          <div class="output-column">
            <div class="output">
              <div
                class="output-class"
                :class="{ predicted: i === predictedClass() }"
                v-for="i in outputClasses"
                :key="`output-class-${i}`"
              >
                <div class="output-label">{{ i }}</div>
                <div
                  class="output-bar"
                  :style="{ width: `${Math.round(180 * output[i])}px` }"
                ></div>
              </div>
            </div>
          </div>
        </v-flex>
      </v-layout>
    </v-container>
  </div>
</template>

<script lang='ts'>
import _ from "lodash";
import { mathUtils, runModelUtils } from "../../utils";

import type { Tensor, InferenceSession } from "onnxruntime-web";
import modelStatus from "./ModelStatus.vue";
import { watch, ref, defineComponent } from "vue";
import type { PropType } from "vue";

export default defineComponent({
  name: "DrawingModelUI",
  components: {
    modelStatus,
  },

  props: {
    modelFilepath: { type: String, required: true },
    preprocess: {
      type: Function as PropType<(ctx: CanvasRenderingContext2D) => Tensor>,
      required: true,
    },
    postprocess: {
      type: Function as PropType<(t: Tensor) => Float32Array>,
      required: true,
    },
    getPredictedClass: {
      type: Function as PropType<(output: Float32Array) => number>,
      required: true,
    },
  },

  setup(props){
    let modelLoading = ref(true);
    let modelInitializing = ref(true);
    let modelLoadingError = ref(false);
    let sessionRunning = ref(false);
    let input: Float32Array = new Float32Array(784);
    let output = ref(new Float32Array(10));
    let outputClasses: number[] = _.range(10);
    let drawing = ref(false);
    let strokes: number[][][] = [];
    let inferenceTime = ref(0);
    let session: InferenceSession | undefined;
    let gpuSession: InferenceSession | undefined;
    let cpuSession: InferenceSession | undefined;
    let sessionBackend = ref("webgl");
    let modelFile: ArrayBuffer = new ArrayBuffer(0);
    let backendSelectList: Array<{ text: string; value: string }> = [
      { text: "GPU-WebGL", value: "webgl" },
      { text: "CPU-WebAssembly", value: "wasm" },
    ];

    init();

    watch(sessionBackend, async (newVal: string): Promise<string> => {
      sessionBackend.value = newVal;
      clear();
      try {
        await initSession();
      } catch (e) {
        modelLoadingError.value = true;
      }
      return newVal;
    });

    async function init() {
      console.log("init");
      const response = await fetch(props.modelFilepath);
      modelFile = await response.arrayBuffer();
      try {
        await initSession();
      } catch (e) {
        sessionBackend.value = "wasm";
      }
    };

    async function initSession() {
      console.log("initSession");
      sessionRunning.value = false;
      modelLoadingError.value = false;
      if (sessionBackend.value === "webgl") {
        if (gpuSession) {
          session = gpuSession;
          return;
        }
        modelLoading.value = true;
        modelInitializing.value = true;
      }
      if (sessionBackend.value === "wasm") {
        if (cpuSession) {
          session = cpuSession;
          return;
        }
        modelLoading.value = true;
        modelInitializing.value = true;
      }

      try {
        if (sessionBackend.value === "webgl") {
          gpuSession = await runModelUtils.createModelGpu(modelFile);
          session = gpuSession;
        } else if (sessionBackend.value === "wasm") {
          cpuSession = await runModelUtils.createModelCpu(modelFile);
          session = cpuSession;
        }
      } catch (e) {
        modelLoading.value = false;
        modelInitializing.value = false;
        if (sessionBackend.value === "webgl") {
          gpuSession = undefined;
        } else {
          cpuSession = undefined;
        }
        throw new Error("Error: Backend not supported. ");
      }
      modelLoading.value = false;
      // warm up session with a sample tensor. Use setTimeout(..., 0) to make it an async execution so
      // that UI update can be done.
      if (sessionBackend.value === "webgl") {
        setTimeout(() => {
          runModelUtils.warmupModel(session!, [1, 1, 28, 28]);
          modelInitializing.value = false;
        }, 0);
      } else {
        await runModelUtils.warmupModel(session!, [1, 1, 28, 28]);
        modelInitializing.value = false;
      }
    }

    async function run() {
      console.log("run");
      if (!drawing.value) {
        return;
      }
      drawing.value = false;
      sessionRunning.value = true;
      const ctx = (
        document.getElementById("input-canvas") as HTMLCanvasElement
      ).getContext("2d") as CanvasRenderingContext2D;
      const tensor = props.preprocess(ctx);
      const [res, time] = await runModelUtils.runModel(session!, tensor);
      output.value = props.postprocess(res);
      console.log(output.value);
      inferenceTime.value = time;
      sessionRunning.value = false;
    }

    function clear() {
      const ctx = (
        document.getElementById("input-canvas") as HTMLCanvasElement
      ).getContext("2d") as CanvasRenderingContext2D;
      ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
      const ctxCenterCrop = (
        document.getElementById("input-canvas-centercrop") as HTMLCanvasElement
      ).getContext("2d") as CanvasRenderingContext2D;
      ctxCenterCrop.clearRect(
        0,
        0,
        ctxCenterCrop.canvas.width,
        ctxCenterCrop.canvas.height
      );
      const ctxScaled = (
        document.getElementById("input-canvas-scaled") as HTMLCanvasElement
      ).getContext("2d") as CanvasRenderingContext2D;
      ctxScaled.clearRect(0, 0, ctxScaled.canvas.width, ctxScaled.canvas.height);
      output.value = new Float32Array(10);
      drawing.value = false;
      strokes = [];
    }

    function draw(e: any) {
      if (!drawing.value) {
        return;
      }
      // disable scrolling behavior when drawing
      e.preventDefault();
      const ctx = (
        document.getElementById("input-canvas") as HTMLCanvasElement
      ).getContext("2d") as CanvasRenderingContext2D;
      ctx.lineWidth = 20;
      ctx.lineJoin = ctx.lineCap = "round";
      ctx.strokeStyle = "#393E46";
      ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
      let points = strokes[strokes.length - 1];
      points.push(mathUtils.getCoordinates(e));
      // draw individual strokes
      for (let s = 0, slen = strokes.length; s < slen; s++) {
        points = strokes[s];
        let p1 = points[0];
        let p2 = points[1];
        ctx.beginPath();
        ctx.moveTo(p1[0], p1[1]);
        // draw points in stroke
        // quadratic bezier curve
        for (let i = 1, len = points.length; i < len; i++) {
          const midpoint = mathUtils.getMidpoint(p1, p2);
          ctx.quadraticCurveTo(p1[0], p1[1], midpoint[0], midpoint[1]);
          p1 = points[i];
          p2 = points[i + 1];
        }
        ctx.lineTo(p1[0], p1[1]);
        ctx.stroke();
      }
    }

    function activateDraw(e: any) {
      if (modelLoading.value ||  modelInitializing.value ||  modelLoadingError.value ) {
        return;
      }
      drawing.value = true;
      strokes.push([]);
      const points = strokes[strokes.length - 1];
      points.push(mathUtils.getCoordinates(e));
      draw(e);
    }

    function predictedClass(): number {
      return props.getPredictedClass(output.value);
    }

    return {
      modelLoading,
      modelInitializing,
      modelLoadingError,
      sessionRunning,
      input,
      output,
      outputClasses,
      drawing,
      strokes,
      inferenceTime,
      session,
      gpuSession,
      cpuSession,
      sessionBackend,
      modelFile,
      backendSelectList,
      activateDraw,
      run,
      draw,
      clear,
      predictedClass,
    }
  }
});
</script>

<style scoped lang="postcss">
@import "../../variables.css";
.image-panel {
  padding: 40px 20px;
  margin-top: 30px;
  background-color: white;
  position: relative;

  & .loading-indicator {
    position: absolute;
    top: 5px;
    left: 5px;
  }

  & .error-message {
    color: var(--color-error);
    font-size: 12px;
    position: absolute;
    top: 5px;
    left: 5px;
  }
}
.input-column {
  /* height: 100%; */
  display: flex;
  align-items: center;
  justify-content: center;
  flex-direction: column;
  & .input-container {
    width: 100%;
    text-align: center;
    margin: 20px;
    position: relative;
    user-select: none;
    & .input-label {
      font-family: var(--font-sans-serif);
      font-size: 18px;
      color: var(--color-lightgray);
      text-align: center;
      & span.arrow {
        font-size: 36px;
        color: #cccccc;
        position: absolute;
        /* right: -32px; */
        top: 8px;
      }
    }
    & .canvas-container {
      display: inline-flex;
      justify-content: flex-end;
      margin: 10px 0;
      border: 15px solid var(--color-blue-lighter);
      transition: border-color 0.2s ease-in;
      &:hover {
        border-color: var(--color-blue-light);
      }
      & canvas {
        background: whitesmoke;
        &:hover {
          cursor: crosshair;
        }
      }
    }
  }
}
.controls-column {
  height: 100%;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: space-between;
  font-family: var(--font-monospace);
  padding-top: 80px;
  & .control {
    width: 100px;
    margin: 10px 0;
  }
}
.output-column {
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: left;
  margin-left: 80px;
  & .output {
    height: 300;
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    justify-content: center;
    user-select: none;
    cursor: default;
    & .output-class {
      display: flex;
      flex-direction: row;
      align-items: center;
      justify-content: center;
      padding: 10px 0;

      & .output-label {
        text-align: right;
        width: 35px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        font-family: var(--font-sans-serif);
        font-size: 15px;
        color: black;
        padding: 0 6px;
        border-right: 6px solid var(--color-blue-lighter);
      }

      & .output-bar {
        height: 16px;
        transition: width 0.2s ease-out;
        background: var(--color-blue-light);
      }

      & .output-value {
        text-align: left;
        margin-left: 20px;
        font-family: var(--font-sans-serif);
        font-size: 20px;
        color: black;
      }
    }
  }
}

.layer-outputs-container {
  position: relative;
  & .bg-line {
    position: absolute;
    z-index: 0;
    top: 0;
    left: 50%;
    background: whitesmoke;
    width: 15px;
    height: 100%;
  }
  & .layer-output {
    position: relative;
    z-index: 1;
    margin: 30px 20px;
    background: whitesmoke;
    border-radius: 10px;
    padding: 20px;
    overflow-x: auto;
    & .layer-output-heading {
      font-size: 1rem;
      color: #999999;
      margin-bottom: 10px;
      display: flex;
      flex-direction: column;
      font-size: 12px;
      & span.layer-class {
        color: var(--color-blue);
        font-size: 14px;
        font-weight: bold;
      }
    }
    & .layer-output-canvas-container {
      display: inline-flex;
      flex-wrap: wrap;
      background: whitesmoke;
      & canvas {
        border: 1px solid lightgray;
        margin: 1px;
      }
    }
  }
}
/* vue transition `fade` */
.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.5s;
}
.fade-enter,
.fade-leave-to {
  opacity: 0;
}
</style>