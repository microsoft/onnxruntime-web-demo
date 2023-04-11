<template>
  <div class="demo home text-xs-center">
    <v-img class="banner" :src="getImageSource()">
      <v-container class="onnx-wrapper" >
        <v-layout class="onnx-page" >
          <v-slot class="onnx">ONNX Runtime Web</v-slot>
          <v-slot class="run-onnx">Run ONNX model in the browser</v-slot>
          <v-slot class="onnx-info"
            >Interactive ML without install and device independent<br />
            Latency of server-client communication reduced<br />
            Privacy and security ensured<br />
            GPU acceleration</v-slot>
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

import { defineComponent } from "vue";
const DEMO_INFO = [
  {
    title: "MobileNet, trained on ImageNet",
    path: "mobilenet",
    imagePath: "./src/assets/mobilenet.png",
  },
  {
    title: "SqueezeNet, trained on ImageNet",
    path: "squeezenet",
    imagePath: "./src/assets/squeezenet.png",
  },
  {
    title: "MNIST",
    path: "MNIST",
    imagePath: "./src/assets/mnist.png",
  },
  {
    title: "YoLo",
    path: "YoLo",
    imagePath: "./src/assets/yolo.png",
  },
  {
    title: "Emotion Recognition",
    path: "emotion",
    imagePath: "./src/assets/emotion.png",
  },
];

export default defineComponent({
  name: 'HomePage',
  setup() {
    let demoInfo: Array<{ title: string; path: string; imagePath: string }> =
      DEMO_INFO;
    
    function getImageSource(): string{
      return '/src/assets/background.png';
    };

    return {
      demoInfo,
      getImageSource,
    };
  },
});
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
  height: 90px;
  width: 100%;
}

.demo-card-heading {
  color: var(--color-blue);
  flex: 100%;
  font-size: 1.3em;
  transition: color 0.2s ease-out;
  text-align: center;
}

.demo-card-image {
  overflow: hidden;
  display: flex;
  justify-content: right;
  object-fit: contain;  
  height: 100%;
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
  font-size: 0.8em;
  margin-top: 3.5rem;
}

.onnx-page {
  height: 100%;
  text-align: center;
  text-justify: auto;
  align-items: center;
  flex-direction: column;
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
