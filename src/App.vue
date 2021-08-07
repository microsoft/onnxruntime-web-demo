<template>
  <div id="app">
    <v-app>
      <v-navigation-drawer v-model="showNav" absolute fixed floating app>
        <main-menu :currentView="currentView"></main-menu>
      </v-navigation-drawer>
      <v-toolbar app dark flat color="primary">
        <v-toolbar-side-icon
          @click.stop="showNav = !showNav"
        ></v-toolbar-side-icon>
        <v-toolbar-title>{{ currentTitle }}</v-toolbar-title>
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
              <v-layout
                column
                justify-center
                align-center
                fill-height
                class="footer-label"
              >
                {{ currentDescription }}
                <a target="_blank" :href="currentLink">{{ currentLink }}</a>
              </v-layout>
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
import Component from "vue-class-component";
import Vue from "vue";

@Component({
  components: { MainMenu },
})
export default class App extends Vue {
  showNav: boolean;
  hasWebGL: boolean;

  constructor() {
    super();
    this.showNav = false;
    this.hasWebGL = true;
  }

  get currentView() {
    const path = this.$route.path;
    return path.replace(/^\//, "") || "home";
  }

  get currentTitle() {
    const title = DEMO_TITLES[this.currentView];
    if (title) {
      return title;
    } else {
      return "ONNX Runtime Web";
    }
  }
  get currentDescription() {
    const description = DEMO_DESCRIPTIONS[this.currentView];
    if (description) {
      return description;
    } else {
      return "";
    }
  }
  get currentLink() {
    const link = DEMO_MODEL_LINKS[this.currentView];
    if (link) {
      return link;
    } else {
      return "";
    }
  }
}
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
  font-size: 16px;
  color: var(--color-lightgray);
  text-align: left;
  user-select: none;
  cursor: default;
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
