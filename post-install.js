const fs = require("fs");
const path = require("path");

// copy onnxruntime-web WebAssembly files to {workspace}/public/ folder
const srcFolder = path.join(__dirname, 'node_modules', 'onnxruntime-web', 'dist');
const destFolder = path.join(__dirname, 'public', 'js');
if (!fs.existsSync(destFolder)) {
    fs.mkdirSync(destFolder);
}
fs.copyFileSync(path.join(srcFolder, 'ort-wasm.wasm'), path.join(destFolder, 'ort-wasm.wasm'));
fs.copyFileSync(path.join(srcFolder, 'ort-wasm-simd.wasm'), path.join(destFolder, 'ort-wasm-simd.wasm'));
fs.copyFileSync(path.join(srcFolder, 'ort-wasm-threaded.wasm'), path.join(destFolder, 'ort-wasm-threaded.wasm'));
fs.copyFileSync(path.join(srcFolder, 'ort-wasm-simd-threaded.wasm'), path.join(destFolder, 'ort-wasm-simd-threaded.wasm'));
fs.copyFileSync(path.join(srcFolder, 'ort.min.js'), path.join(destFolder, 'ort.min.js'));
