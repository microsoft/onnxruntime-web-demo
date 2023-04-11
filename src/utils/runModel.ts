import {InferenceSession, Tensor} from 'onnxruntime-web';

function init() {
  // env.wasm.simd = false;
}

export async function createModelCpu(model: ArrayBuffer): Promise<InferenceSession> {
  console.log("prior init");
  init();
  console.log("prior create wasm");
  return await InferenceSession.create(model, {executionProviders: ['wasm']});
}
export async function createModelGpu(model: ArrayBuffer): Promise<InferenceSession> {
  init();
  console.log("prior create");
  return await InferenceSession.create(model, {executionProviders: ['webgl']});
}

export async function warmupModel(model: InferenceSession, dims: number[]) {
  // OK. we generate a random input and call Session.run() as a warmup query
  const size = dims.reduce((a, b) => a * b);
  const warmupTensor = new Tensor('float32', new Float32Array(size), dims);

  for (let i = 0; i < size; i++) {
    warmupTensor.data[i] = Math.random() * 2.0 - 1.0;  // random value [-1.0, 1.0)
  }
  try {
    const feeds: Record<string, Tensor> = {};
    feeds[model.inputNames[0]] = warmupTensor;
    // console.log("prior run");
    await model.run(feeds);
    // console.log("after run");
  } catch (e) {
    console.error(e);
  }
}

export async function runModel(
  model: InferenceSession | undefined,
  preprocessedData: Tensor
): Promise<[Tensor, number]> {
  const start = new Date();
  try {
    const feeds: Record<string, Tensor> = {};
    feeds[model!.inputNames[0]] = preprocessedData;
    const outputData = await model!.run(feeds);
    const end = new Date();
    const inferenceTime = (end.getTime() - start.getTime());
    const output = outputData[model!.outputNames[0]];

    return [output, inferenceTime];
  } catch (e) {
    console.error(e);
    throw new Error();
  }
}