import { createApp } from "vue";
import App from "./App.vue";
import router from "./router";

// Vuetify
import "vuetify/styles";
import { createVuetify } from "vuetify";
import type { ThemeDefinition } from "vuetify";
import * as components from "vuetify/components";
import * as directives from "vuetify/directives";

const myCustomLightTheme: ThemeDefinition = {
    dark: false,
    colors: {
    primary: "#2a6a96",
    secondary: "#69707a",
    accent: "#f5d76e",
    error: "#d24d57",
    },
};

const vuetify = createVuetify({
  components,
  directives,
  theme: {
    defaultTheme: "myCustomLightTheme",
    themes: {
      myCustomLightTheme,
    }
  }
});

const app = createApp(App);

app.use(router).use(vuetify);

app.mount("#app");
