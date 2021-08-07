import {Tensor} from 'onnxruntime-web';
import {TypedArrayUtil} from './yoloPostprocessUtils';

export function sigmoid(input: Tensor): Tensor {
  const X = input.data;
  const Y = TypedArrayUtil.createTypedArray(input.type, X.length);
  for (let i = 0; i < X.length; i++) {
    Y[i] = (1 / (1 + Math.exp(-X[i] as number)));
  }
  return new Tensor(input.type, Y, input.dims ? input.dims : [input.data.length]);
}

export function exp(input: Tensor): Tensor {
  if (input.type === 'string') {
    throw new Error('Unsupported type for transform');
  }
  const X = input.data;
  const Y = TypedArrayUtil.createTypedArray(input.type, X.length);
  for (let i = 0; i < X.length; i++) {
    Y[i] = Math.exp(X[i] as number);
  }
  return new Tensor(input.type, Y, input.dims ? input.dims : [input.data.length]);
}