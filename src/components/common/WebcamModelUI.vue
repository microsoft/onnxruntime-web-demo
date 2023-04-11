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
        style="align-self: center; margin: auto; width: 50%; margin-left: 25%; padding: 40px;border-radius: 12px;"
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

      <v-layout row wrap justify-space-around class="webcam-panel elevation-1">
        <div class="webcam-container" id="webcam-container" display="none">
          <video playsinline muted id="webcam" width="416" height="416"></video>
          <canvas
            id="input-canvas"
            width="416"
            height="416"
            willReadFrequently="true"
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
            <div style="width: 100%">
              <v-select
                v-model="imageURLSelect"
                :items="imageUrls"
                :disabled="
                  modelLoading ||
                  modelInitializing ||
                  modelLoadingError ||
                  webcamEnabled
                "
                item-title="text"
                item-value="value"
                label="Select image"
                :menu-props="{ maxHeight: '750' }"
                solo
                single-line
                hide-details
              ></v-select>
            </div>
          </div>
          <v-card-text style="align-self: center; justify-content: center; font-size: large; margin-top: 20%;">or</v-card-text>
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
                style="display:none; margin: 0; width: 100%"
                hide-details
                type="file"
                id="input-upload-image"
                @change="handleFileChange"
              />
            </label>
          </div>
          <v-card-text style="align-self: center; justify-content: center; font-size: large; margin-top: 20%;">or</v-card-text>
          <v-btn
            style="margin: 0; width: 100%; border-radius: 12px;"
            v-on:click="webcamController()"
            :disabled="modelLoadingError"
          >
            {{ webcamStatus() }}
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

import type { InferenceSession, Tensor } from "onnxruntime-web";

import {
  watch,
  defineComponent,
  ref,
  nextTick,
  onMounted,
  onBeforeMount,
} from "vue";
import type { PropType } from "vue";

import loadImage from "blueimp-load-image";
import modelStatus from "../common/ModelStatus.vue";
import { runModelUtils } from "../../utils";

// export default class WebcamModelUI extends Vue {
export default defineComponent({
  name:'WebcamModelUI',
  
  components: {
    modelStatus,
  },

  props: {
    hasWebGL: {
      type: Boolean,
      required: true,
    },
    modelFilepath: {
      type: String,
      required: true,
    },
    imageSize: {
      type: Number,
      required: true,
    },
    imageUrls: {
      type: Array as PropType<Array<{ text: string; value: string }>>,
      required: true,
    },
    warmupModel: {
      type: Function as PropType<(session: InferenceSession) => Promise<void>>,
      required: true,
    },
    preprocess: {
      type: Function as PropType<(ctx: CanvasRenderingContext2D) => Tensor>,
      required: true,
    },
    postprocess: {
      type: Function as PropType<(t: Tensor, inferenceTime: number) => void>,
      required: true,
    },
  },

  setup(props, { emit }){
    let webcamElement: HTMLVideoElement | undefined;
    let videoOrigWidth: number = 0;
    let videoOrigHeight: number = 0;
    let webcamContainer: HTMLElement | null = null;
    let inferenceTime = ref(0);
    let session: InferenceSession | undefined;
    let gpuSession: InferenceSession | undefined;
    let cpuSession: InferenceSession | undefined;

    let modelLoading = ref(true);
    let modelInitializing = ref(true);
    let sessionRunning = ref(false);
    let modelLoadingError = ref(false);

    let imageURLInput: string | null = "";
    let imageURLSelect = ref("" as string | null);
    let imageLoading = ref(false);
    let imageLoadingError = ref(false);

    let webcamEnabled = ref(false)
    let webcamInitialized = ref(false);
    let webcamStream: MediaStream = new MediaStream();

    let sessionBackend = ref("webgl");
    let modelFile: ArrayBuffer = new ArrayBuffer(0);
    let backendSelectList: Array<{ text: string; value: string }> = [
      { text: "GPU-WebGL", value: "webgl" },
      { text: "CPU-WebAssembly", value: "wasm" },
    ];

    onMounted(() => {
      webcamElement = document.getElementById("webcam") as HTMLVideoElement;
      webcamContainer = document.getElementById(
        "webcam-container"
      ) as HTMLElement;
    });

    onBeforeMount(() => {
      stopCamera();
      if (webcamInitialized.value) {
        webcamStream.getTracks()[0].stop();
      }
    });

    async function initSession() {
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
          props.warmupModel(session!);
          modelInitializing.value = false;
        }, 0);
      } else {
        await props.warmupModel(session!);
        modelInitializing.value = false;
      }
    }

    async function init() {
      // fetch the model file to be used later
      const response = await fetch(props.modelFilepath);
      modelFile = await response.arrayBuffer();
      try {
        await initSession();
      } catch (e) {
        sessionBackend.value = "wasm";
      }
    }

    function clearRects() {
      console.log("clearRects");
      // console.log("webcam child length -", webcamContainer!.childNodes!.length);
      while (
        webcamContainer !== null &&
        webcamContainer.childNodes!.length > 2
      ) {
        console.log("clearRects - removing child");
        webcamContainer.removeChild(webcamContainer.childNodes[2]);
      }
    }

    function clearFileInput() {
      const file = document.getElementById("input-upload-image") as HTMLInputElement;
      if (file) {
        file.value = '';
      }
    }

    // Capture image from video
    function capture(): CanvasRenderingContext2D {
      const size = Math.min(videoOrigWidth, videoOrigHeight);
      const centerHeight = videoOrigHeight / 2;
      const beginHeight = centerHeight - size / 2;
      const centerWidth = videoOrigWidth / 2;
      const beginWidth = centerWidth - size / 2;

      // placeholder to draw a image
      const canvas = document.getElementById("screenshot") as HTMLCanvasElement;
      canvas.width = Math.min(webcamElement!.width, webcamElement!.height);
      canvas.height = Math.min(webcamElement!.width, webcamElement!.height);
      const context = canvas.getContext("2d") as CanvasRenderingContext2D;
      context.drawImage(
        webcamElement!,
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

    async function stopCamera() {
      if (webcamElement === undefined) return;
      webcamElement.pause();
      while (sessionRunning.value) {
        await new Promise<void>((resolve) =>
          requestAnimationFrame(() => resolve())
        );
      }
      clearRects();
      clearCanvas();
      webcamEnabled.value = false;
    }

    function clearCanvas() {
      inferenceTime.value = 0;
      imageURLInput = "";
      imageURLSelect.value = null;
      imageLoading.value = false;
      imageLoadingError.value = false;
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

    function loadImageToCanvas(url: string | null) {
      if (!url && url === null) {
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
            imageLoadingError.value = true;
            imageLoading.value = false;
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
            imageLoadingError.value = false;
            imageLoading.value = false;
            sessionRunning.value = true;
            inferenceTime.value = 0;
            // model predict
            nextTick(function () {
              setTimeout(() => {
                runModel(ctx);
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

    watch(imageURLSelect, (newURL: string | null) => {
      if (webcamEnabled.value) {
        stopCamera();
      }
      console.log("imageURLSelect watch");
      imageURLInput = newURL;
      clearRects();
      loadImageToCanvas(newURL);
    });

    watch(sessionBackend, async (newVal: string) => {
      sessionBackend.value = newVal;
      if (webcamEnabled.value) {
        stopCamera();
      }
      clearRects();
      clearCanvas();
      clearFileInput();
      try {
        await  initSession();
      } catch (e) {
        modelLoadingError.value = true;
      }
      return newVal;
    });

    function webcamController() {
      if (webcamEnabled.value) {
        stopCamera();
      } else {
        clearRects();
        runLiveVideo();
      }
    }

    async function startCamera() {
      if (!webcamInitialized.value) {
        sessionRunning.value = true;
        try {
          await cameraSetup();
        } catch (e) {
          sessionRunning.value = false;
          webcamEnabled.value = false;
          alert("no webcam found");
          return;
        }
        webcamElement!.play();
        webcamInitialized.value = true;
        sessionRunning.value = false;
      } else {
        await webcamElement!.play();
      }
      webcamEnabled.value = true;
    }

    async function runLiveVideo() {
      await startCamera();
      if (!webcamEnabled.value) {
        return;
      }
      while (webcamEnabled.value) {
        const ctx =  capture();
        // run model
        await runModel(ctx);
        await new Promise<void>((resolve) =>
          requestAnimationFrame(() => resolve())
        );
      }
    }

    async function runModel(ctx: CanvasRenderingContext2D) {
      sessionRunning.value = true;
      const data = props.preprocess(ctx);
      let outputTensor: Tensor;
      [outputTensor,  inferenceTime.value] = await runModelUtils.runModel(
        session,
        data
      );
      clearRects();
      props.postprocess(outputTensor,  inferenceTime.value);
      sessionRunning.value = false;
    }

    /**
     * Adjusts the video size so we can make a centered square crop without
     * including whitespace.
     * @param {number} width The real width of the video element.
     * @param {number} height The real height of the video element.
     */
    function adjustVideoSize(width: number, height: number) {
      if(!webcamElement) return;
      const aspectRatio = width / height;
      if (width >= height) {
        webcamElement.width = aspectRatio *  webcamElement.height;
      } else if (width < height) {
        webcamElement.height =  webcamElement.width / aspectRatio;
      }
    }

    async function cameraSetup() {
      if(!webcamElement) return;
      if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        const stream = await navigator.mediaDevices.getUserMedia({
          audio: false,
          video: { facingMode: "environment" },
        });
        webcamStream = stream;
        webcamElement.srcObject = stream;
        return new Promise<void>((resolve) => {
          webcamElement!.onloadedmetadata = () => {
            videoOrigWidth =  webcamElement!.videoWidth;
            videoOrigHeight =  webcamElement!.videoHeight;
            adjustVideoSize( videoOrigWidth,  videoOrigHeight);
            resolve();
          };
        });
      } else {
        throw new Error("No webcam found!");
      }
    }

    function webcamStatus() {
      if (webcamEnabled.value) {
        return "Stop Camera";
      } else {
        return "Start Camera";
      }
    }

    function handleFileChange(e: any) {
      emit("input", e.target.files[0]);
      loadImageToCanvas(e.target.files[0]);
    }

    init();

    return{
      webcamElement,
      videoOrigWidth,
      videoOrigHeight,
      webcamContainer,
      inferenceTime,
      session,
      gpuSession,
      cpuSession,
      modelLoading,
      modelInitializing,
      sessionRunning,
      modelLoadingError,
      imageURLInput,
      imageURLSelect,
      imageLoading,
      imageLoadingError,
      webcamEnabled,
      webcamInitialized,
      webcamStream,
      sessionBackend,
      modelFile,
      backendSelectList,
      webcamStatus,
      webcamController,
      handleFileChange,
    }
  }, 
});
</script>

<style lang="postcss" scoped>
@import "../../variables.css";

.ui-container {
  font-family: var(--font-sans-serif);
  margin-bottom: 30px;
}
.webcam-panel {
  padding: 40px 20px;
  width: 70%;
  margin-left: 20%;
  margin-top: 30px;
  background-color: white;
  position: relative;
}
.webcam-container {
  border-radius: 5px;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.12), 0 1px 2px rgba(0, 0, 0, 0.24);
  margin: auto;
  width: 416px;
  height: 416px;
  position: relative;
  display: flex;
  align-items: center;
  justify-content: left;
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
  background: #f5f5f5;
  box-shadow: 0 3px 1px -2px rgba(0, 0, 0, 0.2), 0 2px 2px 0 rgba(0, 0, 0, 0.14),
    0 1px 5px 0 rgba(0, 0, 0, 0.12);
  border-radius: 12px;
  display: inline-flex;
  height: 40px;
  width: 180px;
  align-items: center;
  justify-content: center;
  font-size: 14px;
  transition: 0.3s cubic-bezier(0.25, 0.8, 0.5, 1), color 1ms;
  padding: 0 0px;
  margin: 0 20px;
}

.inputs:focus,
.inputs:hover {
  position: relative;
  background: rgba(0, 0, 0, 0.12);
}
</style>
