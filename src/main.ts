import '@babel/polyfill';
import 'whatwg-fetch';
import 'vuetify/dist/vuetify.min.css';

import Vue from 'vue';
import Vuetify from 'vuetify';

import App from './App.vue';
import router from './router';

Vue.config.productionTip = false;
Vue.use(Vuetify, {theme: {primary: '#2a6a96', secondary: '#69707a', accent: '#f5d76e', error: '#d24d57'}});

new Vue({
  router,
  render: (h) => h(App),
}).$mount('#app');
