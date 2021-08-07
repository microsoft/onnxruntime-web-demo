import {Tensor} from 'onnxruntime-web';
import {NumberDataType} from './yoloPostprocess';
import {ShapeUtil, TypedArrayUtil} from './yoloPostprocessUtils';

export function softmax(x: Tensor, axis: number): Tensor {
  const inputDimensions = x.dims ? x.dims : [x.data.length];
  const inputRank = inputDimensions.length;

  const axisCorrected = ShapeUtil.getActualAxisFromNegativeValue(axis, inputRank);
  const N = ShapeUtil.sizeToDimension(inputDimensions, axisCorrected);
  const D = ShapeUtil.sizeFromDimension(inputDimensions, axisCorrected);

  const X = x.data as NumberDataType;

  const Y = TypedArrayUtil.createTypedArray(x.type, x.data.length);

  for (let i = 0; i < N; i++) {
    // find row offset
    const offset = i * D;

    // find max of each logical row
    let max = Number.MIN_VALUE;
    for (let j = 0; j < D; j++) {
      if (X[offset + j] > max) {
        max = X[offset + j];
      }
    }

    // find normalization scale per row
    let scale = 0;
    for (let j = 0; j < D; j++) {
      const value = X[offset + j] - max;
      Y[offset + j] = Math.exp(value);
      scale += Math.exp(value);
    }

    // perform the softmax normalization
    for (let j = 0; j < D; j++) {
      if (scale === 0) {
        Y[offset + j] = 0;
      } else {
        Y[offset + j] /= scale;
      }
    }
  }

  return new Tensor(x.type, Y, inputDimensions);
}