<template>
  <div id="app">
    <v-app>
      <v-navigation-drawer v-model="showNav" absolute fixed floating app>
        <main-menu :currentView="currentView()"></main-menu>
      </v-navigation-drawer>
      <v-toolbar app dark flat color="primary">
        <v-toolbar-side-icon
          @click.stop="showNav = !showNav"
        ></v-toolbar-side-icon>
        <v-toolbar-title>{{ currentTitle() }}</v-toolbar-title>
        <v-spacer></v-spacer>
      </v-toolbar>

      <v-content>
        <v-container
          @click.stop="showNav = false"
          fluid
          fill-height
          class="content-panel"
        >
          <div class="demo">
            <div class="ui-container">
              <router-view :hasWebGL="hasWebGL"></router-view>
              <v-layout column align-center fill-height class="footer-label">
                {{ currentDescription() }}
              </v-layout>
              <a
                column
                align-center
                fill-height
                target="_blank"
                class="model-link"
                :href="currentLink()"
                >{{ currentLink() }}</a
              >
            </div>
          </div>
        </v-container>
      </v-content>
    </v-app>
  </div>
</template>

<script lang="ts">
import MainMenu from "./components/MainMenu.vue";
import {
  DEMO_TITLES,
  DEMO_DESCRIPTIONS,
  DEMO_MODEL_LINKS,
} from "./data/demo-titles";

import { defineComponent } from "vue";
import { useRouter } from "vue-router";

export default defineComponent({
  components: { MainMenu },
  setup() {
    let showNav: boolean = false;
    let hasWebGL: boolean = true;

    const $route = useRouter();

    function currentView() {
      const path = $route.currentRoute.value.path;
      console.log(path);
      return path.replace(/^\//, "") || "home";
    }

    function currentTitle() {
      const title = DEMO_TITLES[currentView()];
      if (title) {
        return title;
      } else {
        return "ONNX Runtime Web";
      }
    }

    function currentDescription() {
      const description = DEMO_DESCRIPTIONS[currentView()];
      if (description) {
        return description;
      } else {
        return "";
      }
    }

    function currentLink() {
      const link = DEMO_MODEL_LINKS[currentView()];
      if (link) {
        return link;
      } else {
        return "";
      }
    }

    return {
      showNav,
      hasWebGL,
      currentView,
      currentTitle,
      currentDescription,
      currentLink,
    };
  },
});
</script>

<style lang="postcss">
@import "./variables.css";

.application {
  font-family: var(--font-sans-serif) !important;
  font-size: 18px;
}

.application.theme--light {
  background: linear-gradient(0deg, #cccccc, #f0f0f0) !important;
  color: var(--color-darkgray);
}

footer {
  background: #cccccc !important;
}

.footer-label {
  font-family: var(--font-sans-serif);
  font-size: 10px;
  color: var(--color-lightgray);
  text-align: center;
  user-select: none;
  cursor: default;
  width: 40%;
  margin: 0 25% 0 25%;
}

.model-link {
  font-family: var(--font-sans-serif);
  font-size: 10px;
  text-align: center;
  user-select: none;
  cursor: default;
  width: 40%;
  margin: 0 25% 0 25%;
}

a {
  text-decoration: none;
}

.demo {
  position: relative;
  width: 100%;
  height: 100%;
}

/*******************************************************************/
/* Vuetify overrides */

.navigation-drawer {
  background-color: whitesmoke !important;
}

.input-group--select .input-group__selections__comma,
.input-group input,
.input-group textarea {
  font-size: 20px !important;
  color: var(--color-black) !important;
}

.input-group:not(.input-group--error) label {
  font-size: 20px !important;
  color: var(--color-lightgray) !important;
}

.list .list__tile:not(.list__tile--active) {
  color: var(--color-darkgray) !important;
}

.list__tile {
  font-size: 16px !important;
  height: 35px !important;
  font-family: var(--font-monospace);
}

.content-panel {
  padding: 0 !important;
}

.select-backend {
  text-align: center;
  font-family: var(--font-sans-serif);
  font-size: 20px;
  color: var(--color-black);
  margin-right: 10px;
}

.error-message {
  color: var(--color-error);
  font-size: 15px;
  text-align: center;
}
</style>
