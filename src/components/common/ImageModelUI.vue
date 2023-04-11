<template>
  <div>
    <!-- session Loading and Initializing Indicator -->
    <model-status
      v-if="modelLoading || modelInitializing"
      :modelLoading="modelLoading"
      :modelInitializing="modelInitializing"
    ></model-status>
    <v-container fluid
      style="margin-left: 20%; width: 60%; padding: 30px"
    >
      <!-- Utility bar to select session backend configs. -->
      <v-layout
        align-center
        style="margin-left: 10%; width: 70%; padding: 30px"
      >
        <div class="select-backend" style="align-self: center;">Select Backend:</div>
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
      <v-layout row wrap justify-center class="image-panel elevation-1">
        <!-- model status -->
        <div v-if="imageLoading || sessionRunning" class="loading-indicator">
          <v-progress-circular indeterminate color="primary" />
        </div>
        <!-- select input images -->
        <v-flex sm6 md4 align-center justify-center column fill-height>
          <v-layout align-center>
            <v-flex sm4>
              <v-select
                v-model="imageURLSelect"
                :disabled="
                  modelLoading || modelInitializing || modelLoadingError
                "
                :items="imageURLSelectList"
                item-title="text"
                item-value="value"
                label="Select image"
                :menu-props="{ maxHeight: '750' }"
                solo
                single-line
                hide-details
              ></v-select>
            </v-flex>
            <v-flex class="text-xs-center"> or </v-flex>
            <label
              :disabled="modelLoading || modelInitializing || modelLoadingError"
              class="inputs"
            >
              <div>
                <span>UPLOAD IMAGE</span>
              </div>
              <input
                style="display: none"
                type="file"
                id="input-upload-image"
                @change="handleFileChange"
              />
            </label>
          </v-layout>
          <!-- input image -->
          <div
            v-if="imageLoadingError"
            class="error-message"
            style="padding-top: 30px"
          >
            Error loading URL
          </div>
          <v-flex align-center justify-space-between class="canvas-container">
            <canvas
              id="input-canvas"
              :width="imageSize"
              :height="imageSize"
            ></canvas>
          </v-flex>
        </v-flex>

        <v-flex sm6 md4 column fill-height class="output-container">
          <v-flex class="inference-time-class">
            <span class="inference-time">Inference Time: </span>
            <span v-if="inferenceTime > 0" class="inference-time-value"
              >{{ inferenceTime.toFixed(1) }} ms
            </span>
            <span v-else>-</span>
          </v-flex>
          <div
            v-for="i in [0, 1, 2, 3, 4]"
            :key="i"
            class="output-class"
            :class="{
              predicted: i === 0 && outputClasses.value[i].probability > 0,
            }"
          >
            <div class="output-label">{{ outputClasses.value[i].name }}</div>
            <div
              class="output-bar"
              :style="{
                width: `${Math.round(180 * outputClasses.value[i].probability)}px`,
                background: `rgba(42, 106, 150, ${outputClasses.value[
                  i
                ].probability.toFixed(2)})`,
                transition: `${
                  outputClasses.value[i].probability != 0
                    ? 'width 0.2s ease-out'
                    : 'null'
                }`,
              }"
            ></div>
            <div class="output-value">
              {{ Math.round(100 * outputClasses.value[i].probability) }}%
            </div>
          </div>
        </v-flex>
      </v-layout>
    </v-container>
  </div>
</template>

<script lang="ts">

import loadImage from "blueimp-load-image";
import { runModelUtils } from "../../utils";
import type { ClassResult } from "../../utils/imagenet";

import modelStatus from "./ModelStatus.vue";
import type { InferenceSession, Tensor } from "onnxruntime-web";
import { watch, defineComponent, ref, reactive, nextTick } from "vue";
import type { PropType } from "vue";

export default defineComponent({
  name:'ImageClassification',
  props: {
    modelFilepath: { type: String, required: true },
    imageSize: { type: Number, required: true },
    imageUrls: {
      type: Array as PropType<Array<{ text: string; value: string }>>,
      required: true,
    },
    preprocess: {
      type: Function as PropType<(ctx: CanvasRenderingContext2D) => Tensor>,
      required: true,
    },
    getPredictedClass: {
      type: Function as PropType<(output: Float32Array) => ClassResult[]>,
      required: true,
    },
  },

  components: {
    modelStatus,
  },

  beforeMount() {
    this.session = undefined;
    this.gpuSession = undefined;
    this.cpuSession = undefined;
  },

  setup(props, { emit }) {
    let sessionBackend = ref("webgl");
    let currentStatus = ref("Started setup function");
    let backendSelectList: Array<{ text: string; value: string }> = 
    [
      { text: "GPU-WebGL", value: "webgl" },
      { text: "CPU-WebAssembly", value: "wasm" },
    ];
    let modelLoading = ref(true);
    let modelInitializing = ref(true);
    let modelLoadingError = ref(false);
    let sessionRunning = ref(false);
    let session: InferenceSession | undefined;
    let gpuSession: InferenceSession | undefined;
    let cpuSession: InferenceSession | undefined;

    let inferenceTime = ref(0);
    let imageURLInput: string = "";
    const imageURLSelect = ref("" as string | null);
    const imageURLSelectList = ref(props.imageUrls);
    let imageLoading = ref(false);
    let imageLoadingError = ref(false);
    let output: Float32Array = new Float32Array(0);
    let modelFile: ArrayBuffer = new ArrayBuffer(0);
    let outputClasses = reactive(
      {value: props.getPredictedClass(output) as ClassResult[]}
    );

    async function initSession() {
      currentStatus.value = "entered initSession function";
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
      currentStatus.value = "modelLoading -" + modelLoading.value + " - modelInitializing -" + modelInitializing.value + " - sessionBackend.value -" + sessionBackend.value;

      try {
        if (sessionBackend.value === "webgl") {
          currentStatus.value = "entered webgl session";
          gpuSession = await runModelUtils.createModelGpu(modelFile);
          session = gpuSession;
          currentStatus.value = "webgl session set";
        } else if (sessionBackend.value === "wasm") {
          currentStatus.value = "entered wasm session";
          console.log("entered wasm session");
          cpuSession = await runModelUtils.createModelCpu(modelFile);
          console.log("cpuSession - " + cpuSession);
          session = cpuSession;
          currentStatus.value = "wasm session set";
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
      currentStatus.value = "session created";
      // warm up session with a sample tensor. Use setTimeout(..., 0) to make it an async execution so
      // that UI update can be done.
      if (sessionBackend.value === "webgl") {
        setTimeout(() => {
          currentStatus.value = "session warming up - 1";
          currentStatus.value = "model file -" + modelFile + " - sessionBackend.value -" + sessionBackend.value + " - file - " + props.modelFilepath;
          runModelUtils.warmupModel(session!, [
            1,
            3,
            props.imageSize,
            props.imageSize,
          ]);
          modelInitializing.value = false;
        }, 0);
        currentStatus.value = "session warming up - 2";
      } else {
        await runModelUtils.warmupModel(session!, [
          1,
          3,
          props.imageSize,
          props.imageSize,
        ]);
        currentStatus.value = "session warmed up";
        modelInitializing.value = false;
      }
    }

    function loadImageToCanvas(url: string) {
      if (!url) {
        clearAll();
        return;
      }
      imageLoading.value = true;
      loadImage(
        url,
        (img: Event | HTMLImageElement | HTMLCanvasElement) => {
          if ((img as Event).type === "error") {
            imageLoadingError.value = true;
            imageLoading.value = false;
          } else {
            // load image data onto input canvas
            const element = document.getElementById(
              "input-canvas"
            ) as HTMLCanvasElement;
            // console.log("loadImageToCanvas - element - " + element);
            if (element) {
              const ctx = element.getContext("2d");
              // console.log("loadImageToCanvas - ctx - " + ctx);
              if (ctx) {
                ctx.drawImage(img as HTMLImageElement, 0, 0);
                imageLoadingError.value = false;
                imageLoading.value = false;
                sessionRunning.value = true;
                output = new Float32Array(0);
                inferenceTime.value = 0;
                // session predict
                nextTick(() => {
                  setTimeout(() => {
                    runModel(ctx);
                  }, 10);
                });
              }
            }
          }
        },
        {
          maxWidth: props.imageSize,
          maxHeight: props.imageSize,
          cover: true,
          crop: true,
          canvas: true,
          crossOrigin: "Anonymous",
        }
      );
    }

    async function runModel(ctx: CanvasRenderingContext2D) {
      const preprocessedData = props.preprocess(ctx);
      let tensorOutput = null;
      [tensorOutput, inferenceTime.value] = await runModelUtils.runModel(
        session!,
        preprocessedData
      );
      output = tensorOutput.data as Float32Array;
      outputClasses.value = props.getPredictedClass(output);
      sessionRunning.value = false;
    }

    function clearAll() {
      sessionRunning.value = false;
      inferenceTime.value = 0;
      imageURLInput = "";
      imageURLSelect.value = null;
      imageLoading.value = false;
      imageLoadingError.value = false;
      output = new Float32Array(0);

      const element = document.getElementById(
        "input-canvas"
      ) as HTMLCanvasElement;
      if (element) {
        const ctx = element.getContext("2d");
        if (ctx) {
          ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
        }
      }

      const file = document.getElementById(
        "input-upload-image"
      ) as HTMLInputElement;
      if (file) {
        file.value = "";
      }
    }

    function onImageURLInputEnter(e: any) {
      imageURLSelect.value = null;
      loadImageToCanvas(e.target.value);
    }

    function handleFileChange(e: any) {
      emit("input", e.target.files[0]);
      loadImageToCanvas(e.target.files[0]);
    }

    watch(sessionBackend, async (newVal: string) => {
      sessionBackend.value = newVal;
      clearAll();
      try {
        await initSession();
      } catch (e) {
        modelLoadingError.value = true;
      }
      return newVal;
    });

    watch(imageURLSelect, (newVal: string | null) => {
      if (newVal === null) return;
      imageURLInput = newVal;
      loadImageToCanvas(newVal);
    });

    async function setupSession() {
      // fetch the model file to be used later
      currentStatus.value = "Loading model file...";
      const response = await fetch(props.modelFilepath);
      currentStatus.value = "Model file loaded.";
      modelFile = await response.arrayBuffer();
      currentStatus.value = "generating model buffer";
      try {
        await initSession();
      } catch (e) {
        sessionBackend.value = "wasm";
      }
    }

    setupSession();

    return {
      sessionBackend,
      backendSelectList,
      modelLoading,
      modelInitializing,
      modelLoadingError,
      sessionRunning,
      session,
      gpuSession,
      cpuSession,
      inferenceTime,
      imageURLInput,
      imageURLSelect,
      imageURLSelectList,
      imageLoading,
      imageLoadingError,
      output,
      modelFile,
      handleFileChange,
      onImageURLInputEnter,
      outputClasses,
      currentStatus,
    };
  },
});
</script>

<style lang="postcss" scoped>
@import "../../variables.css";

.text-xs-center {
  text-align: center;
  white-space: nowrap;
  font-family: var(--font-sans-serif);
  font-size: 16px;
  color: black;
  padding: 10px 10px;
}
.image-panel {
  padding: 80px 0px 80px 0px;
  margin: auto;
  background-color: white;
  position: relative;
  width: 85%;
  height: 100%;
  & .loading-indicator {
    position: absolute;
    top: 5px;
    left: 5px;
  }
}

.inputs {
  margin: auto;
  background: #f5f5f5;
  box-shadow: 0 3px 1px -2px rgba(0, 0, 0, 0.2), 0 2px 2px 0 rgba(0, 0, 0, 0.14),
    0 1px 5px 0 rgba(0, 0, 0, 0.12);
  align-items: center;
  border-radius: 2px;
  display: inline-flex;
  height: 40px;
  font-size: 14px;
  transition: 0.3s cubic-bezier(0.25, 0.8, 0.5, 1), color 1ms;
  padding: 0 16px;
}

.inputs:focus,
.inputs:hover {
  position: relative;
  background: rgba(0, 0, 0, 0.12);
}

.input-label {
  font-family: var(--font-sans-serif);
  font-size: 16px;
  color: var(--color-blue);
  text-align: left;
  user-select: none;
  cursor: default;
}

.canvas-container {
  position: relative;
  text-align: center;
  & #input-canvas {
    background: #eeeeee;
    margin-top: 40px;
  }
}

.output-container {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  justify-content: center;

  & .inference-time-class {
    display: flex;
    flex-direction: row;
    align-items: center;
    justify-content: center;
    & .inference-time {
      text-align: right;
      width: 200px;
      white-space: nowrap;
      text-overflow: ellipsis;
      font-family: var(--font-sans-serif);
      font-size: 20px;
      color: black;
    }

    & .inference-time-value {
      color: var(--color-blue);
      text-align: left;
      margin-left: 20px;
      font-family: var(--font-sans-serif);
      font-size: 20px;
    }
  }

  & .output-class {
    display: flex;
    flex-direction: row;
    align-items: center;
    justify-content: center;
    padding: 5px 0;
    margin-top: 20px;

    & .output-label {
      text-align: right;
      width: 200px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      font-family: var(--font-sans-serif);
      font-size: 20px;
      color: black;
      padding: 0 16px;
      border-right: 6px solid var(--color-blue-lighter);
    }

    & .output-bar {
      height: 20px;
      transition: width 0.2s ease-out;
      color: var(--color-blue-light);
    }

    & .output-value {
      text-align: left;
      margin-left: 20px;
      font-family: var(--font-sans-serif);
      font-size: 20px;
      color: black;
    }
  }

  & .output-class.predicted {
    & .output-label {
      color: var(--color-blue);
      border-left-color: var(--color-blue);
    }

    & .output-value {
      color: var(--color-blue);
    }
  }
}
</style>
