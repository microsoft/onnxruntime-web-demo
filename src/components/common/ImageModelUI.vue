<template>
  <div>
    <!-- session Loading and Initializing Indicator -->
    <model-status
      v-if="modelLoading || modelInitializing"
      :modelLoading="modelLoading"
      :modelInitializing="modelInitializing"
    ></model-status>
    <v-container fluid>
      <!-- Utility bar to select session backend configs. -->
      <v-layout
        justify-center
        align-center
        style="margin: auto; width: 40%; padding: 30px"
      >
        <div class="select-backend">Select Backend:</div>
        <v-select
          v-model="sessionBackend"
          :disabled="modelLoading || modelInitializing || sessionRunning"
          :items="backendSelectList"
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

      <v-layout row wrap justify-space-around class="image-panel elevation-1">
        <!-- model status -->
        <div v-if="imageLoading || sessionRunning" class="loading-indicator">
          <v-progress-circular indeterminate color="primary" />
        </div>
        <!-- select input images -->
        <v-flex sm6 md4 align-center justify-start column fill-height>
          <v-layout align-center>
            <v-flex sm4>
              <v-select
                v-model="imageURLSelect"
                :disabled="
                  modelLoading || modelInitializing || modelLoadingError
                "
                :items="imageURLSelectList"
                label="Select image"
                :menu-props="{ maxHeight: '750' }"
                solo
                single-line
                hide-details
              ></v-select>
            </v-flex>
            <v-flex class="text-xs-center">or</v-flex>
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
              predicted: i === 0 && outputClasses[i].probability.toFixed(2) > 0,
            }"
          >
            <div class="output-label">{{ outputClasses[i].name }}</div>
            <div
              class="output-bar"
              :style="{
                width: `${Math.round(180 * outputClasses[i].probability)}px`,
                background: `rgba(42, 106, 150, ${outputClasses[
                  i
                ].probability.toFixed(2)})`,
                transition: `${
                  outputClasses[i].probability != 0
                    ? 'width 0.2s ease-out'
                    : 'null'
                }`,
              }"
            ></div>
            <div class="output-value">
              {{ Math.round(100 * outputClasses[i].probability) }}%
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

import modelStatus from "./ModelStatus.vue";
import { InferenceSession, Tensor } from "onnxruntime-web";
import { Vue, Component, Prop, Watch } from "vue-property-decorator";

@Component({
  components: {
    modelStatus,
  },
})
export default class ImageModelUI extends Vue {
  @Prop({ type: String, required: true }) modelFilepath!: string;
  @Prop({ type: Number, required: true }) imageSize!: number;
  @Prop({ type: Array, required: true }) imageUrls!: Array<{
    text: string;
    value: string;
  }>;
  @Prop({ type: Function, required: true }) preprocess!: (
    ctx: CanvasRenderingContext2D
  ) => Tensor;
  @Prop({ type: Function, required: true }) getPredictedClass!: (
    output: Float32Array
  ) => {};

  sessionBackend: string;
  backendSelectList: Array<{ text: string; value: string }>;
  modelLoading: boolean;
  modelInitializing: boolean;
  modelLoadingError: boolean;
  sessionRunning: boolean;
  session: InferenceSession | undefined;
  gpuSession: InferenceSession | undefined;
  cpuSession: InferenceSession | undefined;

  inferenceTime: number;
  imageURLInput: string;
  imageURLSelect: null;
  imageURLSelectList: Array<{ text: string; value: string }>;
  imageLoading: boolean;
  imageLoadingError: boolean;
  output: Tensor.DataType;
  modelFile: ArrayBuffer;

  constructor() {
    super();
    this.sessionBackend = "webgl";
    this.backendSelectList = [
      { text: "GPU-WebGL", value: "webgl" },
      { text: "CPU-WebAssembly", value: "wasm" },
    ];
    this.modelLoading = true;
    this.modelInitializing = true;
    this.modelLoadingError = false;
    this.sessionRunning = false;
    this.inferenceTime = 0;
    this.imageURLInput = "";
    this.imageURLSelect = null;
    this.imageURLSelectList = this.imageUrls;
    this.imageLoading = false;
    this.imageLoadingError = false;
    this.output = [];
    this.modelFile = new ArrayBuffer(0);
  }

  async created() {
    // fetch the model file to be used later
    const response = await fetch(this.modelFilepath);
    this.modelFile = await response.arrayBuffer();
    try {
      await this.initSession();
    } catch (e) {
      this.sessionBackend = "wasm";
    }
  }

  async initSession() {
    this.sessionRunning = false;
    this.modelLoadingError = false;
    if (this.sessionBackend === "webgl") {
      if (this.gpuSession) {
        this.session = this.gpuSession;
        return;
      }
      this.modelLoading = true;
      this.modelInitializing = true;
    }
    if (this.sessionBackend === "wasm") {
      if (this.cpuSession) {
        this.session = this.cpuSession;
        return;
      }
      this.modelLoading = true;
      this.modelInitializing = true;
    }

    try {
      if (this.sessionBackend === "webgl") {
        this.gpuSession = await runModelUtils.createModelGpu(this.modelFile);
        this.session = this.gpuSession;
      } else if (this.sessionBackend === "wasm") {
        this.cpuSession = await runModelUtils.createModelCpu(this.modelFile);
        this.session = this.cpuSession;
      }
    } catch (e) {
      this.modelLoading = false;
      this.modelInitializing = false;
      if (this.sessionBackend === "webgl") {
        this.gpuSession = undefined;
      } else {
        this.cpuSession = undefined;
      }
      throw new Error("Error: Backend not supported. ");
    }
    this.modelLoading = false;
    // warm up session with a sample tensor. Use setTimeout(..., 0) to make it an async execution so
    // that UI update can be done.
    if (this.sessionBackend === "webgl") {
      setTimeout(() => {
        runModelUtils.warmupModel(this.session!, [
          1,
          3,
          this.imageSize,
          this.imageSize,
        ]);
        this.modelInitializing = false;
      }, 0);
    } else {
      await runModelUtils.warmupModel(this.session!, [
        1,
        3,
        this.imageSize,
        this.imageSize,
      ]);
      this.modelInitializing = false;
    }
  }

  @Watch("sessionBackend")
  async onSessionBackendChange(newVal: string) {
    this.sessionBackend = newVal;
    this.clearAll();
    try {
      await this.initSession();
    } catch (e) {
      this.modelLoadingError = true;
    }
    return newVal;
  }

  @Watch("imageURLSelect")
  onImageURLSelectChange(newVal: string) {
    this.imageURLInput = newVal;
    this.loadImageToCanvas(newVal);
  }

  beforeDestroy() {
    this.session = undefined;
    this.gpuSession = undefined;
    this.cpuSession = undefined;
  }

  get outputClasses() {
    return this.getPredictedClass(Array.prototype.slice.call(this.output));
  }

  onImageURLInputEnter(e: any) {
    this.imageURLSelect = null;
    this.loadImageToCanvas(e.target.value);
  }

  handleFileChange(e: any) {
    this.$emit("input", e.target.files[0]);
    this.loadImageToCanvas(e.target.files[0]);
  }

  loadImageToCanvas(url: string) {
    if (!url) {
      this.clearAll();
      return;
    }
    this.imageLoading = true;
    loadImage(
      url,
      (img) => {
        if ((img as Event).type === "error") {
          this.imageLoadingError = true;
          this.imageLoading = false;
        } else {
          // load image data onto input canvas
          const element = document.getElementById(
            "input-canvas"
          ) as HTMLCanvasElement;
          if (element) {
            const ctx = element.getContext("2d");
            if (ctx) {
              ctx.drawImage(img as HTMLImageElement, 0, 0);
              this.imageLoadingError = false;
              this.imageLoading = false;
              this.sessionRunning = true;
              this.output = [];
              this.inferenceTime = 0;
              // session predict
              this.$nextTick(function () {
                setTimeout(() => {
                  this.runModel();
                }, 10);
              });
            }
          }
        }
      },
      {
        maxWidth: this.imageSize,
        maxHeight: this.imageSize,
        cover: true,
        crop: true,
        canvas: true,
        crossOrigin: "Anonymous",
      }
    );
  }

  async runModel() {
    const element = document.getElementById(
      "input-canvas"
    ) as HTMLCanvasElement;
    const ctx = element.getContext("2d") as CanvasRenderingContext2D;
    const preprocessedData = this.preprocess(ctx);
    let tensorOutput = null;
    [tensorOutput, this.inferenceTime] = await runModelUtils.runModel(
      this.session!,
      preprocessedData
    );
    this.output = tensorOutput.data;
    this.sessionRunning = false;
  }

  clearAll() {
    this.sessionRunning = false;
    this.inferenceTime = 0;
    this.imageURLInput = "";
    this.imageURLSelect = null;
    this.imageLoading = false;
    this.imageLoadingError = false;
    this.output = [];

    const element = document.getElementById(
      "input-canvas"
    ) as HTMLCanvasElement;
    if (element) {
      const ctx = element.getContext("2d");
      if (ctx) {
        ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
      }
    }

    const file = document.getElementById("input-upload-image") as HTMLInputElement;
    if (file) {
      file.value = '';
    }
  }
}
</script>

<style lang="postcss" scoped>
@import "../../variables.css";
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
      height: 16px;
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
