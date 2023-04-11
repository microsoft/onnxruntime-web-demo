import { createRouter, createWebHistory } from "vue-router";

import Home from "../components/Home.vue";
import EmotionRecognition from "../components/models/Emotion.vue";
import MNIST from "../components/models/MNIST.vue";
import MobileNet from "../components/models/Mobilenet.vue";
import SqueezeNet from "../components/models/Squeezenet.vue";
import YoLo from "../components/models/Yolo.vue";

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: '/',
      name: 'home',
      component: Home,
    },
    {
      path: '/mobilenet',
      name: 'mobilenet',
      component: MobileNet,
    },
    {
      path: '/squeezenet',
      name: 'squeezenet',
      component: SqueezeNet,
    },
    {
      path: '/MNIST',
      name: 'MNIST',
      component: MNIST,
    },
    {
      path: '/YoLo',
      name: 'YoLo',
      component: YoLo,
    },
    {
      path: '/emotion',
      name: 'EmotionRecognition',
      component: EmotionRecognition,
    },
  ],
});

export default router;
