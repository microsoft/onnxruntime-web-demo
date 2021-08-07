import {Tensor} from 'onnxruntime-web';
import {ShapeUtil, TypedArrayUtil} from './yoloPostprocessUtils';

export function reshape(x: Tensor, shape: ReadonlyArray<number>): Tensor {
  const reshapedDims = ShapeUtil.calculateReshapedDims(x.dims, shape);
  const output = new Tensor(x.type, TypedArrayUtil.createTypedArray(x.type, x.data.length), reshapedDims);
  const X = x.data;
  const Y = output.data;
  for (let i = 0; i < x.data.length; ++i) {
    Y[i] = X[i];
  }
  return output;
}
