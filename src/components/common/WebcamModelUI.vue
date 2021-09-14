<template>
  <div>
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
        style="margin: auto; width: 40%; padding: 40px"
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

      <v-layout row wrap justify-space-around class="webcam-panel elevation-1">
        <div class="webcam-container" id="webcam-container" display="none">
          <video playsinline muted id="webcam" width="416" height="416"></video>
          <canvas
            id="input-canvas"
            width="416"
            height="416"
            style="position: absolute"
            v-show="!webcamEnabled"
          ></canvas>
        </div>
        <v-progress-circular
          v-show="sessionRunning"
          indeterminate
          color="primary"
          height="250px"
        />

        <v-flex
          justify-center
          align-center
          sm6
          class="text-xs-center"
          style="display: flex; flex-direction: column"
        >
          <div
            class="text-xs-center"
            style="display: flex; justify-content: center"
          >
            <div v-if="imageLoadingError" class="error-message">
              Error loading URL
            </div>
            <div style="width: 70%">
              <v-select
                v-model="imageURLSelect"
                :items="imageUrls"
                :disabled="
                  modelLoading ||
                  modelInitializing ||
                  modelLoadingError ||
                  webcamEnabled
                "
                label="Select image"
                :menu-props="{ maxHeight: '750' }"
                solo
                single-line
                hide-details
              ></v-select>
            </div>
          </div>
          <v-card-text>or</v-card-text>
          <div
            :disabled="
              modelLoading ||
              modelInitializing ||
              modelLoadingError ||
              webcamEnabled
            "
            style="margin: 0; width: 30%"
          >
            <label class="inputs">
              UPLOAD IMAGE
              <input
                style="display: none"
                type="file"
                id="input-upload-image"
                @change="handleFileChange"
              />
            </label>
          </div>
          <v-card-text>or</v-card-text>

          <v-btn
            style="margin: 0; width: 30%"
            v-on:click="webcamController"
            :disabled="modelLoadingError"
          >
            {{ webcamStatus }}
          </v-btn>
        </v-flex>
      </v-layout>
    </v-container>
    <canvas id="screenshot" v-show="false"></canvas>
  </div>
</template>

<script lang="ts">
/**
 * - setup()
 * - capture()
 * - adjustVideoSize()
 * are adapted from:
 * https://github.com/ModelDepot/tfjs-yolo-tiny-demo/blob/master/src/webcam.js
 */

import { InferenceSession, Tensor } from "onnxruntime-web";
import { Vue, Component, Prop, Watch } from "vue-property-decorator";
import loadImage from "blueimp-load-image";
import ModelStatus from "../common/ModelStatus.vue";
import { runModelUtils } from "../../utils";

@Component({
  components: {
    ModelStatus,
  },
})
export default class WebcamModelUI extends Vue {
  @Prop(Boolean) hasWebGL!: boolean;
  @Prop({ type: String, required: true }) modelFilepath!: string;
  @Prop({ type: Number, required: true }) imageSize!: number;
  @Prop({ type: Array, required: true }) imageUrls!: Array<{
    text: string;
    value: string;
  }>;
  @Prop({ type: Function, required: true }) warmupModel!: (
    session: InferenceSession
  ) => Promise<void>;
  @Prop({ type: Function, required: true }) preprocess!: (
    ctx: CanvasRenderingContext2D
  ) => Tensor;
  @Prop({ type: Function, required: true }) postprocess!: (
    t: Tensor,
    inferenceTime: number
  ) => void;

  webcamElement: HTMLVideoElement;
  videoOrigWidth: number;
  videoOrigHeight: number;
  webcamContainer: HTMLElement;
  inferenceTime: number;
  session: InferenceSession;
  gpuSession: InferenceSession | undefined;
  cpuSession: InferenceSession | undefined;

  modelLoading: boolean;
  modelInitializing: boolean;
  sessionRunning: boolean;
  modelLoadingError: boolean;

  imageURLInput: string;
  imageURLSelect: null;
  imageURLSelectList: Array<{ text: string; value: string }>;
  imageLoading: boolean;
  imageLoadingError: boolean;

  webcamEnabled: boolean;
  webcamInitialized: boolean;
  webcamStream: MediaStream;

  sessionBackend: string;
  modelFile: ArrayBuffer;
  backendSelectList: Array<{ text: string; value: string }>;

  constructor() {
    super();
    this.inferenceTime = 0;
    this.imageURLInput = "";
    this.imageURLSelect = null;
    this.imageURLSelectList = this.imageUrls;
    this.imageLoading = false;
    this.imageLoadingError = false;

    this.modelLoading = true;
    this.modelInitializing = true;
    this.sessionRunning = false;
    this.modelLoadingError = false;

    this.webcamEnabled = false;
    this.webcamInitialized = false;

    this.sessionBackend = "webgl";
    this.modelFile = new ArrayBuffer(0);
    this.backendSelectList = [
      { text: "GPU-WebGL", value: "webgl" },
      { text: "CPU-WebAssembly", value: "wasm" },
    ];
  }

  async mounted() {
    this.webcamElement = document.getElementById("webcam") as HTMLVideoElement;
    this.webcamContainer = document.getElementById(
      "webcam-container"
    ) as HTMLElement;
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
        this.warmupModel(this.session!);
        this.modelInitializing = false;
      }, 0);
    } else {
      await this.warmupModel(this.session!);
      this.modelInitializing = false;
    }
  }

  @Watch("sessionBackend")
  async onSessionBackendChange(newVal: string) {
    this.sessionBackend = newVal;
    if (this.webcamEnabled) {
      this.stopCamera();
    }
    this.clearRects();
    this.clearCanvas();
    this.clearFileInput();
    try {
      await this.initSession();
    } catch (e) {
      this.modelLoadingError = true;
    }
    return newVal;
  }

  @Watch("imageURLSelect")
  onImageURLSelectChange(newVal: string) {
    if (this.webcamEnabled) {
      this.stopCamera();
    }
    this.imageURLInput = newVal;
    this.clearRects();
    this.loadImageToCanvas(newVal);
  }

  handleFileChange(e: any) {
    this.$emit("input", e.target.files[0]);
    this.loadImageToCanvas(e.target.files[0]);
  }

  get webcamStatus() {
    if (this.webcamEnabled) {
      return "Stop Camera";
    } else {
      return "Start Camera";
    }
  }

  beforeDestroy() {
    this.stopCamera();
    if (this.webcamInitialized) {
      this.webcamStream.getTracks()[0].stop();
    }
  }

  webcamController() {
    if (this.webcamEnabled) {
      this.stopCamera();
    } else {
      this.clearRects();
      this.runLiveVideo();
    }
  }

  loadImageToCanvas(url: string) {
    if (!url) {
      const element = document.getElementById(
        "input-canvas"
      ) as HTMLCanvasElement;
      const ctx = element.getContext("2d") as CanvasRenderingContext2D;
      ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
      return;
    }
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
          const ctx = element.getContext("2d") as CanvasRenderingContext2D;
          const imageWidth = (img as HTMLImageElement).width;
          const imageHeight = (img as HTMLImageElement).height;
          ctx.drawImage(
            img as HTMLImageElement,
            0,
            0,
            imageWidth,
            imageHeight,
            0,
            0,
            element.width,
            element.height
          );
          this.imageLoadingError = false;
          this.imageLoading = false;
          this.sessionRunning = true;
          this.inferenceTime = 0;
          // model predict
          this.$nextTick(function () {
            setTimeout(() => {
              this.runModel(ctx);
            }, 10);
          });
        }
      },
      {
        cover: true,
        crop: true,
        canvas: true,
        crossOrigin: "Anonymous",
      }
    );
  }

  async setup() {
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: { facingMode: "environment" },
      });
      this.webcamStream = stream;
      this.webcamElement.srcObject = stream;
      return new Promise<void>((resolve) => {
        this.webcamElement.onloadedmetadata = () => {
          this.videoOrigWidth = this.webcamElement.videoWidth;
          this.videoOrigHeight = this.webcamElement.videoHeight;
          this.adjustVideoSize(this.videoOrigWidth, this.videoOrigHeight);
          resolve();
        };
      });
    } else {
      throw new Error("No webcam found!");
    }
  }

  async stopCamera() {
    this.webcamElement.pause();
    while (this.sessionRunning) {
      await new Promise<void>((resolve) =>
        requestAnimationFrame(() => resolve())
      );
    }
    this.clearRects();
    this.clearCanvas();
    this.webcamEnabled = false;
  }

  async startCamera() {
    if (!this.webcamInitialized) {
      this.sessionRunning = true;
      try {
        await this.setup();
      } catch (e) {
        this.sessionRunning = false;
        this.webcamEnabled = false;
        alert("no webcam found");
        return;
      }
      this.webcamElement.play();
      this.webcamInitialized = true;
      this.sessionRunning = false;
    } else {
      await this.webcamElement.play();
    }
    this.webcamEnabled = true;
  }

  async runLiveVideo() {
    await this.startCamera();
    if (!this.webcamEnabled) {
      return;
    }
    while (this.webcamEnabled) {
      const ctx = this.capture();
      // run model
      await this.runModel(ctx);
      await new Promise<void>((resolve) =>
        requestAnimationFrame(() => resolve())
      );
    }
  }

  async runModel(ctx: CanvasRenderingContext2D) {
    this.sessionRunning = true;
    const data = this.preprocess(ctx);
    let outputTensor: Tensor;
    [outputTensor, this.inferenceTime] = await runModelUtils.runModel(
      this.session,
      data
    );
    this.clearRects();
    this.postprocess(outputTensor, this.inferenceTime);
    this.sessionRunning = false;
  }

  clearCanvas() {
    this.inferenceTime = 0;
    this.imageURLInput = "";
    this.imageURLSelect = null;
    this.imageLoading = false;
    this.imageLoadingError = false;
    const element = document.getElementById(
      "input-canvas"
    ) as HTMLCanvasElement;
    if (element) {
      const ctx = element.getContext("2d");
      if (ctx) {
        ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
      }
    }
  }

  clearRects() {
    while (this.webcamContainer.childNodes.length > 2) {
      this.webcamContainer.removeChild(this.webcamContainer.childNodes[2]);
    }
  }

  clearFileInput() {
    const file = document.getElementById("input-upload-image") as HTMLInputElement;
    if (file) {
      file.value = '';
    }
  }

  // Capture image from video
  capture(): CanvasRenderingContext2D {
    const size = Math.min(this.videoOrigWidth, this.videoOrigHeight);
    const centerHeight = this.videoOrigHeight / 2;
    const beginHeight = centerHeight - size / 2;
    const centerWidth = this.videoOrigWidth / 2;
    const beginWidth = centerWidth - size / 2;

    // placeholder to draw a image
    const canvas = document.getElementById("screenshot") as HTMLCanvasElement;
    canvas.width = Math.min(
      this.webcamElement.width,
      this.webcamElement.height
    );
    canvas.height = Math.min(
      this.webcamElement.width,
      this.webcamElement.height
    );
    const context = canvas.getContext("2d") as CanvasRenderingContext2D;
    context.drawImage(
      this.webcamElement,
      beginWidth,
      beginHeight,
      size,
      size,
      0,
      0,
      canvas.width,
      canvas.height
    );
    return context;
  }
  /**
   * Adjusts the video size so we can make a centered square crop without
   * including whitespace.
   * @param {number} width The real width of the video element.
   * @param {number} height The real height of the video element.
   */
  adjustVideoSize(width: number, height: number) {
    const aspectRatio = width / height;
    if (width >= height) {
      this.webcamElement.width = aspectRatio * this.webcamElement.height;
    } else if (width < height) {
      this.webcamElement.height = this.webcamElement.width / aspectRatio;
    }
  }
}
</script>

<style lang="postcss" scoped>
@import "../../variables.css";

.ui-container {
  font-family: var(--font-sans-serif);
  margin-bottom: 30px;
}
.webcam-panel {
  padding: 40px 20px;
  margin-top: 30px;
  background-color: white;
  position: relative;
}
.webcam-container {
  border-radius: 5px;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.12), 0 1px 2px rgba(0, 0, 0, 0.24);
  margin: 0 auto;
  width: 416px;
  height: 416px;
  position: relative;
  display: flex;
  align-items: center;
  justify-content: center;
  overflow: hidden;
  & :nth-child(n + 3) {
    position: absolute;
    border: 1px solid red;
    font-size: 24px;
    & :first-child {
      background: white;
      color: black;
      opacity: 0.8;
      font-size: 12px;
      padding: 3px;
      text-transform: capitalize;
      white-space: nowrap;
    }
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
  width: 100%;
  height: 38px;
  font-size: 14px;
  transition: 0.3s cubic-bezier(0.25, 0.8, 0.5, 1), color 1ms;
  justify-content: center;
  padding: 0 16px;
}

.inputs:focus,
.inputs:hover {
  position: relative;
  background: rgba(0, 0, 0, 0.12);
}
</style>
