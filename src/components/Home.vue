<template>
  <div class="demo home text-xs-center">
    <v-img class="banner" :src="require('@/assets/background.png')">
      <v-container class="onnx-wrapper">
        <v-layout column justify-center align-center>
          <v-flex class="onnx">ONNX Runtime Web</v-flex>
          <v-flex class="run-onnx">Run ONNX model in the browser</v-flex>
          <v-flex class="onnx-info"
            >Interactive ML without install and device independent<br />
            Latency of server-client communication reduced<br />
            Privacy and security ensured<br />
            GPU acceleration</v-flex
          >
        </v-layout>
      </v-container>
    </v-img>
    <div class="demo-card-wrapper">
      <div v-for="info in demoInfo" :key="info.path" class="demo-card">
        <router-link :to="`/${info.path}`">
          <div class="demo-card-image"><img :src="info.imagePath" /></div>
          <div class="demo-card-heading">{{ info.title }}</div>
        </router-link>
      </div>
    </div>
  </div>
</template>

<script lang="ts">
import { Component, Vue } from "vue-property-decorator";
const DEMO_INFO = [
  {
    title: "SqueezeNet, trained on ImageNet",
    path: "squeezenet",
    imagePath: require("@/assets/squeezenet.png"),
  },
  {
    title: "ResNet50, trained on ImageNet",
    path: "resnet50",
    imagePath: require("@/assets/resnet50.png"),
  },
  {
    title: "Emotion FerPlus",
    path: "emotion_ferplus",
    imagePath: require("@/assets/emotion.png"),
  },
  { title: "Yolo", path: "yolo", imagePath: require("@/assets/yolo.png") },
  { title: "MNIST", path: "mnist", imagePath: require("@/assets/mnist.png") },
];

@Component
export default class HomePage extends Vue {
  demoInfo: Array<{ title: string; path: string; imagePath: string }> =
    DEMO_INFO;

  constructor() {
    super();
    this.demoInfo = DEMO_INFO;
  }
}
</script>

<style scoped lang="postcss">
@import "../variables.css";

.home {
  background: var(--color-blue);
  height: 100%;
  width: 100%;
}

.demo-card-wrapper {
  margin-top: 2rem;
  display: flex;
  flex-direction: column;
  align-items: center;
}

.demo-card {
  width: 90%;
  max-width: 1000px;
  font-family: var(--font-sans-serif);
  height: 90px;
  background: white;
  border: 1px solid whitesmoke;
  cursor: default;
  user-select: none;
  box-shadow: 3px 3px #062d5b;
  transition: box-shadow 0.2s ease-out;
  margin-bottom: 1rem;

  &:first-child {
    margin-top: 0;
  }

  &:hover {
    box-shadow: 3px 3px 5px var(--color-blue-light);
    cursor: pointer;

    & .demo-card-heading {
      color: var(--color-blue);
    }
  }
}

.demo-card a {
  display: flex;
  align-items: center;
}

.demo-card-heading {
  color: var(--color-lightgray);
  flex: 1;
  font-size: 1.1em;
  transition: color 0.2s ease-out;
  text-align: center;
}

.demo-card-image {
  height: 90px;

  & img {
    width: auto;
    height: 100%;
  }
}

.banner {
  color: white;
  height: 33rem;
}

.onnx-wrapper {
  margin-top: 5rem;
}

.onnx {
  font-size: 3em;
}

.run-onnx {
  font-size: 1.5em;
}

.onnx-info {
  font-family: var(--font-sans-serif-regular);
  font-size: 1em;
  margin-top: 5rem;
}

@media (max-width: 500px) {
  .banner {
    height: 10rem;
  }

  .onnx-wrapper {
    padding: 0;
    margin: 0;
    background-color: rgba(0, 0, 0, 0.5);
  }

  .onnx {
    display: none;
  }

  .run-onnx {
    font-size: 1em;
    margin-top: 0.5rem;
  }

  .onnx-info {
    margin-top: 0.5rem;
    font-size: 0.8em;
  }

  .demo-card {
    height: auto;
  }

  .demo-card a {
    flex-direction: column;
  }

  .demo-card-heading {
    margin: 1rem 0;
  }

  .demo-card-image {
    width: 100%;
    height: auto;
  }

  .demo-card-image img {
    width: 100%;
    height: auto;
  }
}
</style>

