import {Tensor} from 'onnxruntime-web';

import {binaryOp} from './binary-op';
import {concat as concatImpl} from './concat';
import {reshape as reshapeImpl} from './reshape';
import {softmax as softmaxImpl} from './softmax';
import {transpose as transposeImpl} from './transpose';
import * as unaryOps from './unary-op';
import {ShapeUtil, TypedArrayUtil, TypeUtil} from './yoloPostprocessUtils';

// Types
export type Type = Tensor.Type;  //'string'|'int32'|'float32'|'bool';
export type NumberType = 'int32'|'float32';
export type NumberOrBoolType = 'int32'|'float32'|'bool';
export type NumberDataType = Uint8Array|Int32Array|Float32Array;

// Utility Tensor Creators
export function as1D(t: Tensor): Tensor {
  return reshape(t, [t.data.length]);
}

export function scalar(value: number, dtype: NumberType = 'float32'): Tensor {
  if (dtype !== 'float32' && dtype !== 'int32') {
    throw new Error('Unsupported type for this transformation');
  }
  const data = TypedArrayUtil.createTypedArray(dtype, 1);
  data[0] = value;
  return new Tensor(dtype, data, [1]);
}

export function zeros(dims: ReadonlyArray<number>, dtype: NumberType = 'float32'): Tensor {
  if (dtype !== 'float32' && dtype !== 'int32' && dtype !== 'bool') {
    throw new Error('Unsupported type for creating all zero Tensor');
  }
  ShapeUtil.validateDims(dims);
  return new Tensor(dtype, TypedArrayUtil.createTypedArray(dtype, ShapeUtil.size(dims)), dims);
}

export function linspace(start: number, stop: number, num: number): Tensor {
  if (num === 0) {
    throw new Error('Must request atleast one sample');
  }
  const increments = (stop - start) / (num - 1);
  const data = TypedArrayUtil.createTypedArray('float32', num);
  data[0] = start;
  for (let i = 1; i < data.length; i++) {
    data[i] = data[i - 1] + increments;
  }
  return new Tensor('float32', data, [num]);
}

export function range(start: number, stop: number, step = 1, dtype: NumberType = 'float32'): Tensor {
  if (step === 0) {
    throw new Error('Step size of 0 is not acceptable');
  }
  // adjust default values
  if (stop < step && step === 1) {
    step = -1;
  }
  // the following conditions cannot generate any data
  if (start === step || (start < stop && step < 0) || (stop < start && step > 0)) {
    return new Tensor(dtype, TypedArrayUtil.createTypedArray(dtype, 1), [1]);
  }
  const size = Math.abs(Math.ceil((stop - start) / step));
  const data = TypedArrayUtil.createTypedArray(dtype, size);
  data[0] = start;
  for (let i = 1; i < data.length; i++) {
    data[i] = data[i - 1] + step;
  }
  return new Tensor(dtype, data, [size]);
}

// Basic Math Tensor Transforms
export function sigmoid(t: Tensor): Tensor {
  if (t.type !== 'float32' && t.type !== 'int32') {
    throw new Error('Unsupported type for transform');
  }
  return unaryOps.sigmoid(t);
}

export function exp(t: Tensor): Tensor {
  if (t.type !== 'float32' && t.type !== 'int32') {
    throw new Error('Unsupported type for transform');
  }
  return unaryOps.exp(t);
}

// Arithmetic Tensor Transforms
export function add(t1: Tensor, t2: Tensor): Tensor {
  if ((t1.type !== 'float32' && t1.type !== 'int32') || (t2.type !== 'float32' && t2.type !== 'int32')) {
    throw new Error('Unsupported type for transform');
  }
  if (t1.type !== t2.type) {
    throw new Error('Types are not homogeneous');
  }
  return binaryOp(t1, t2, (e1, e2) => (e1 + e2), t1.type);
}

export function sub(t1: Tensor, t2: Tensor): Tensor {
  if ((t1.type !== 'float32' && t1.type !== 'int32') || (t2.type !== 'float32' && t2.type !== 'int32')) {
    throw new Error('Unsupported type for transform');
  }
  if (t1.type !== t2.type) {
    throw new Error('Types are not homogeneous');
  }
  return binaryOp(t1, t2, (e1, e2) => (e1 - e2), t1.type);
}

export function mul(t1: Tensor, t2: Tensor): Tensor {
  if ((t1.type !== 'float32' && t1.type !== 'int32') || (t2.type !== 'float32' && t2.type !== 'int32')) {
    throw new Error('Unsupported type for transform');
  }
  if (t1.type !== t2.type) {
    throw new Error('Types are not homogeneous');
  }
  return binaryOp(t1, t2, (e1, e2) => (e1 * e2), t1.type);
}

export function div(t1: Tensor, t2: Tensor): Tensor {
  if ((t1.type !== 'float32' && t1.type !== 'int32') || (t2.type !== 'float32' && t2.type !== 'int32')) {
    throw new Error('Unsupported type for transform');
  }
  if (t1.type !== t2.type) {
    throw new Error('Types are not homogeneous');
  }
  // TODO: Handle division by zero if any
  return binaryOp(t1, t2, (e1, e2) => (e1 / e2), t1.type);
}

// Normalization Tensor Transforms
export function softmax(t: Tensor, dim = -1): Tensor {
  if (t.type !== 'float32' && t.type !== 'int32') {
    throw new Error('Unsupported type for transform');
  }
  return softmaxImpl(t, dim);
}

// Slice And Join Tensor Transforms
export function concat(tensors: Tensor[], axis = 0, typeCheckRequired = true): Tensor {
  if (tensors.length < 2) {
    throw new Error('Must have atleast 2 tensors to concatenate');
  }

  if (typeCheckRequired) {
    const types: Type[] = [];
    tensors.forEach(t => {
      types.push(t.type);
    });
    TypeUtil.validateSameTypes(types);
  }

  return concatImpl(tensors, axis);
}

export function stack(tensors: Tensor[], axis = 0): Tensor {
  if (tensors.length < 2) {
    throw new Error('Must have atleast 2 tensors to stack');
  }

  const types: Type[] = [];
  const shapes: Array<ReadonlyArray<number>> = [];
  tensors.forEach(t => {
    types.push(t.type);
    shapes.push(t.dims ? t.dims : [t.data.length]);
  });
  TypeUtil.validateSameTypes(types);
  ShapeUtil.validateEqualDims(shapes);
  const rank = tensors[0].dims ? tensors[0].dims.length : 1;
  axis = ShapeUtil.getActualAxisFromNegativeValue(axis, rank);
  const expanded = tensors.map(t => expandDims(t, axis));
  return concat(expanded, axis, false);
}

export function gather(t: Tensor, indices: Tensor, axis = 0): Tensor {
  if (t.type === 'string') {
    throw new Error('Unspported type for this transformation');
  }
  if (indices.type !== 'int32' || (indices.dims && indices.dims.length > 1)) {
    throw new Error('Indices tensor not of specified format');
  }
  const dims = t.dims ? t.dims.slice() : [t.data.length];
  const newDims = dims;
  const indicesData = indices.data;
  newDims[axis] = indicesData.length;
  const dimsStrides = ShapeUtil.computeStrides(dims);
  const newDimsStrides = ShapeUtil.computeStrides(newDims);
  const Y = TypedArrayUtil.createTypedArray(t.type, ShapeUtil.size(newDims));
  const X = t.data;
  for (let i = 0; i < Y.length; ++i) {
    const newLogicalIndex = ShapeUtil.offsetToIndices(i, newDimsStrides);
    const oldLogicalIndex = newLogicalIndex.slice();
    oldLogicalIndex[axis] = indicesData[newLogicalIndex[axis]] as number;
    const oldOffset = ShapeUtil.indicesToOffset(oldLogicalIndex, dimsStrides);
    Y[i] = X[oldOffset] as number;
  }
  return new Tensor(t.type, Y, newDims);
}

export function slice(t: Tensor, begin: number[], size: number[]): Tensor {
  if (t.type === 'string') {
    throw new Error('Unspported type for this transformation');
  }
  const newDimsStride = ShapeUtil.computeStrides(size);
  const oldDimsStride = ShapeUtil.computeStrides(t.dims ? t.dims : [t.data.length]);
  const X = t.data;
  const Y = TypedArrayUtil.createTypedArray(t.type, ShapeUtil.size(size));
  for (let i = 0; i < Y.length; ++i) {
    const newLogicalIndex = ShapeUtil.offsetToIndices(i, newDimsStride);
    const oldLogicalIndex = newLogicalIndex.map((idx, j) => idx + begin[j]);
    const oldOffset = ShapeUtil.indicesToOffset(oldLogicalIndex, oldDimsStride);
    Y[i] = X[oldOffset] as number;
  }
  return new Tensor(t.type, Y, size);
}

export function tile(t: Tensor, reps: ReadonlyArray<number>): Tensor {
  if (t.type === 'string') {
    throw new Error('Unspported type for this transformation');
  }
  const dims = t.dims ? t.dims : [t.data.length];
  const rank = dims.length;
  const newDims = new Array(rank);
  if (rank !== reps.length) {
    throw new Error('Repetitions must be of the same rank as input dims');
  }
  for (let i = 0; i < rank; i++) {
    newDims[i] = dims[i] * reps[i];
  }
  const dimsStrides = ShapeUtil.computeStrides(dims);
  const newDimsStrides = ShapeUtil.computeStrides(newDims);
  const Y = TypedArrayUtil.createTypedArray(t.type, ShapeUtil.size(newDims));
  const X = t.data;
  for (let i = 0; i < Y.length; ++i) {
    const newLogicalIndex = ShapeUtil.offsetToIndices(i, newDimsStrides);
    const oldLogicalIndex = new Array(rank);
    for (let j = 0; j < rank; ++j) {
      oldLogicalIndex[j] = newLogicalIndex[j] % t.dims[j];
    }
    const oldOffset = ShapeUtil.indicesToOffset(oldLogicalIndex, dimsStrides);
    Y[i] = X[oldOffset] as number;
  }
  return new Tensor(t.type, Y, newDims);
}

// Permutation Tensor Transforms
export function transpose(t: Tensor, perm?: number[]): Tensor {
  return transposeImpl(t, perm);
}

// Shape Tensor Transforms
export function expandDims(t: Tensor, axis = 0): Tensor {
  axis = ShapeUtil.getActualAxisFromNegativeValue(axis, t.dims ? t.dims.length : 1);
  const dims = t.dims ? t.dims : [t.data.length];
  const changedShapeLength = dims.length + 1;
  const changedShape = new Array<number>(changedShapeLength);
  let iter = 0;
  for (let i = 0; i < changedShapeLength; ++i) {
    if (i === axis) {
      changedShape[i] = 1;
    } else {
      changedShape[i] = dims[iter++];
    }
  }
  return new Tensor(t.type, t.data, changedShape);
}

// Logical Tensor Transforms
export function greaterEqual(t1: Tensor, t2: Tensor): Tensor {
  if ((t1.type !== 'float32' && t1.type !== 'int32' && t1.type !== 'bool') ||
      (t2.type !== 'float32' && t2.type !== 'int32' && t2.type !== 'bool')) {
    throw new Error('Unsupported type for transform');
  }
  if (t1.type !== t2.type) {
    throw new Error('Types are not homogeneous');
  }
  return binaryOp(t1, t2, (e1, e2) => (e1 >= e2 ? 1 : 0), 'bool');
}

export function where(condition: Tensor, t1: Tensor, t2: Tensor): Tensor {
  // validate shape and types of input tensors and condition tensor
  ShapeUtil.areEqual(t1.dims ? t1.dims : [t1.data.length], t2.dims ? t2.dims : [t2.data.length]);
  TypeUtil.validateSameTypes([t1.type, t2.type]);
  if (condition.type !== 'bool') {
    throw new Error('Condition tensor must be bool type');
  }

  // create output
  const outputShape = t1.dims ? t1.dims : [t1.data.length];
  const output =
      new Tensor(t1.type, TypedArrayUtil.createTypedArray(t1.type, ShapeUtil.size(outputShape)), outputShape);
  const outputData = output.data;

  // input data
  const conditionData = condition.data;
  const X = t1.data;
  const Y = t2.data;

  // condition is 1D rank
  if (!condition.dims || condition.dims.length === 1) {
    // the outermost dimension of the input tensors and condition tensor must be the same
    const conditionDims = condition.dims ? condition.dims : [condition.data.length];
    const t1Dims = t1.dims ? t1.dims : [t1.data.length];
    if (conditionDims[0] !== t1Dims[0]) {
      throw new Error('Outermost dimensions of input tensors and condition tensor must match');
    }

    let offset = 1;
    // Input tensors are not 1-D. Need to compute offset.
    if (t1.dims && t1.dims.length > 1) {
      for (let i = 1; i < t1.dims.length; ++i) {
        offset *= t1.dims[i];
      }
    }

    for (let i = 0; i < conditionData.length; ++i) {
      for (let j = 0; j < offset; ++j) {
        outputData[i * offset + j] = conditionData[i] > 0 ? X[i * offset + j] : Y[i * offset + j];
      }
    }
  } else {
    // The shapes of input tensors and condition tensor must be the same
    ShapeUtil.areEqual(condition.dims, t2.dims ? t2.dims : [t2.data.length]);

    for (let i = 0; i < conditionData.length; ++i) {
      outputData[i] = conditionData[i] > 0 ? X[i] : Y[i];
    }
  }
  return output;
}

// Cast Tensor Transforms
export function cast(t: Tensor, dtype: Type): Tensor {
  // TODO: If the requested type and the given type are the same, return same tensor ?
  // Need to investigate if it breaks some basic assumptions
  switch (dtype) {
    case 'int32':
      return new Tensor('int32', Int32Array.from(t.data as NumberDataType), t.dims ? t.dims : [t.data.length]);
    case 'float32':
      return new Tensor('float32', Float32Array.from(t.data as NumberDataType), t.dims ? t.dims : [t.data.length]);
    case 'bool':
      return new Tensor('bool', Uint8Array.from(t.data as NumberDataType), t.dims ? t.dims : [t.data.length]);
    default:
      throw new Error('Unsupported type for casting');
  }
}

export function reshape(t: Tensor, dims: ReadonlyArray<number>): Tensor {
  return reshapeImpl(t, dims);
}

// Reduction Tensor Transforms
export function argMax(t: Tensor, axis = 0): Tensor {
  if (t.type !== 'float32' && t.type !== 'int32') {
    throw new Error('Unsupported type for transform');
  }
  const rank = t.dims ? t.dims.length : 1;
  axis = ShapeUtil.getActualAxisFromNegativeValue(axis, rank);
  const [reduceDims, resultDims] = ShapeUtil.splitDimsIntoTwo(t.dims ? t.dims : [t.data.length], axis);
  const X = t.data;
  const Y = TypedArrayUtil.createTypedArray('int32', resultDims.length === 0 ? 1 : ShapeUtil.size(resultDims));
  const blockSize = reduceDims[0];
  for (let i = 0; i < Y.length; ++i) {
    const offset = blockSize * i;
    let max = X[offset];
    let index = 0;
    for (let j = 0; j < blockSize; ++j) {
      const value = X[offset + j];
      if (value > max) {
        max = value;
        index = j;
      }
    }
    Y[i] = index;
  }
  return new Tensor('int32', Y, resultDims.length === 0 ? [1] : resultDims);
}

export function max(t: Tensor, axis = 0, keepDims = false): Tensor {
  if (t.type !== 'float32' && t.type !== 'int32') {
    throw new Error('Unsupported type for transform');
  }
  const rank = t.dims ? t.dims.length : 1;
  axis = ShapeUtil.getActualAxisFromNegativeValue(axis, rank);
  const [reduceDims, resultDims] = ShapeUtil.splitDimsIntoTwo(t.dims ? t.dims : [t.data.length], axis);
  const X = t.data as NumberDataType;
  const Y = TypedArrayUtil.createTypedArray(t.type, resultDims.length === 0 ? 1 : ShapeUtil.size(resultDims));
  const blockSize = reduceDims[0];
  for (let i = 0; i < Y.length; ++i) {
    const offset = blockSize * i;
    let max = X[offset];
    for (let j = 0; j < blockSize; ++j) {
      const value = X[offset + j];
      if (value > max) {
        max = value;
      }
    }
    Y[i] = max;
  }

  let adjustedResultDims: number[] = [];
  if (keepDims) {
    const origDims = t.dims ? t.dims : [t.data.length];
    for (let i = 0; i < origDims.length; ++i) {
      if (i === axis) {
        adjustedResultDims.push(1);
      } else {
        adjustedResultDims.push(origDims[i]);
      }
    }
  } else {
    adjustedResultDims = resultDims;
  }
  return new Tensor(t.type, Y, adjustedResultDims.length === 0 ? [1] : adjustedResultDims);
}
