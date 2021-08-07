module.exports = {
  css: {
    loaderOptions: {
      postcss: {
        plugins: [
          require('postcss-import')(),
          require('postcss-cssnext')({ browsers: ['>0.5%'] })]
      }
    }
  },
  publicPath: process.env.NODE_ENV === 'production' ? '/onnxruntime-web-demo/' : '/',
  outputDir: 'docs',
  configureWebpack: config => {
    if (process.env.NODE_ENV === 'production') {
      config.node = {
        __dirname: false,
        __filename: false
      }
    }
  }
}

