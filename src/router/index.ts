import Vue from 'vue';
import Router from 'vue-router';

import Home from '../components/Home.vue';
import Emotion from '../components/models/Emotion.vue';
import MNIST from '../components/models/MNIST.vue';
import MobileNet from '../components/models/Mobilenet.vue';
import SqueezeNet from '../components/models/Squeezenet.vue';
import Yolo from '../components/models/Yolo.vue';

Vue.use(Router);

export default new Router({
  mode: 'hash',
  base: process.env.BASE_URL,
  routes: [
    {
      path: '*',
      name: 'home',
      component: Home,
    },
    {
      path: '/mobilenet',
      component: MobileNet,
    },
    {
      path: '/squeezenet',
      component: SqueezeNet,
    },
    {
      path: '/emotion_ferplus',
      component: Emotion,
    },
    {
      path: '/yolo',
      component: Yolo,
    },
    {
      path: '/mnist',
      component: MNIST,
    }
  ],
});
