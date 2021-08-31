<template>
  <aside class="menu">
    <div class="logo"><a href=".">ONNX Runtime Web</a></div>
    <p class="menu-label">Demos</p>
    <ul class="menu-list">
      <li
        v-for="info in demoInfo"
        :key="info.path"
        :class="{ active: currentView === 'mobilenet' }"
      >
        <router-link :to="`/${info.path}`">
          <span class="menu-item-heading">{{ info.model }}</span>
        </router-link>
      </li>
    </ul>
    <p class="menu-label">Links</p>
    <ul class="menu-list github">
      <li>
        <a
          href="https://github.com/microsoft/onnxruntime/tree/master/js/web#readme"
          target="_blank"
          rel="noopener noreferrer"
        >
          <span class="menu-item-heading"> ONNX Runtime Web GitHub</span>
        </a>
      </li>
      <li>
        <a
          href="https://github.com/Microsoft/onnxruntime-web-demo"
          target="_blank"
          rel="noopener noreferrer"
        >
          <span class="menu-item-heading"> ONNX Runtime Web demo Github</span>
        </a>
      </li>
      <li>
        <a href="https://onnx.ai/" target="_blank" rel="noopener noreferrer">
          <span class="menu-item-heading"> ONNX</span>
        </a>
      </li>
    </ul>
  </aside>
</template>

<script scoped lang='ts'>
import { Vue, Component, Prop } from "vue-property-decorator";
const DEMO_INFO = [
  {
    model: "MobileNet",
    title: "MobileNet, trained on ImageNet",
    path: "mobilenet",
  },
  {
    model: "SqueezeNet",
    title: "SqueezeNet, trained on ImageNet",
    path: "squeezenet",
  },
  {
    model: "Emotion FerPlus",
    title: "Emotion FerPlus",
    path: "emotion_ferplus",
  },
  { model: "Yolo", title: "Yolo", path: "yolo" },
  { model: "MNIST", title: "MNIST", path: "mnist" },
];

@Component
export default class MainMenu extends Vue {
  @Prop({ default: "home" }) currentView: string;
  demoInfo: Array<{ model: string; title: string; path: string }> = DEMO_INFO;
  constructor() {
    super();
    this.demoInfo = DEMO_INFO;
  }
}
</script>

<style lang="postcss">
@import "../variables.css";

.menu {
  font-family: var(--font-sans-serif);
  padding: 20px 40px;
  background: whitesmoke;
}

.logo {
  font-size: 20px;

  & img {
    max-width: 100%;
    max-height: 100%;
  }
}

.menu-label {
  user-select: none;
  cursor: default;
  font-size: 11px;
  color: var(--color-lightgray);
  letter-spacing: 2px;
  text-transform: uppercase;
  margin: 11px 0;
}

.menu-list {
  list-style: none;

  & li {
    color: var(--color-lightgray);
    border-left: 2px solid whitesmoke;
    margin-bottom: 5px;
  }

  & li.active {
    border-left: 2px solid var(--color-blue);
  }

  & a {
    padding: 7px 11px;
    color: var(--color-blue);
    text-decoration: none;
    display: flex;
    flex-direction: row;
    align-items: center;
    justify-content: space-between;

    &:hover {
      color: var(--color-blue);
      background-color: whitesmoke;
    }

    & span.menu-item-heading {
      display: flex;
      align-items: center;
      margin-right: 10px;
      font-size: 14px;
    }

    & span.menu-item-subheading {
      color: #999999;
      font-size: 10px;
    }
  }
}

.menu-list.github,
.menu-list.contact {
  & li {
    padding: 5px 10px;
    font-size: 14px;
  }

  & a {
    color: var(--color-blue);
    padding: 0;
    display: inline-flex;
    background-color: none;
    transition: color 0.2s ease-in-out;

    & .icon {
      color: var(--color-blue);
      transition: color 0.2s ease-in-out;
      margin-right: 5px;
    }

    &:hover {
      color: var(--color-blue-light);
      background: none;

      & .icon {
        color: var(--color-blue-light);
      }
    }
  }
}
</style>
