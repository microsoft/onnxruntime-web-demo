
import {Tensor} from 'onnxruntime-web';
import {NumberDataType} from './yoloPostprocess';
import {arrayCopyHelper, ShapeUtil, TypedArrayUtil} from './yoloPostprocessUtils';

export function transpose(x: Tensor, perm?: number[]): Tensor {
  const inputDims = x.dims ? x.dims : [x.data.length];
  const rank = inputDims.length;

  // determine permutation to use
  // if no permutation was specified in the attributes,
  // the default is [rank-1, ..., 0]
  let finalPerm = new Array<number>(rank);
  if (perm && perm.length === rank) {
    finalPerm = perm;
  } else {
    for (let i = 0; i < rank; i++) {
      finalPerm[i] = rank - i - 1;
    }
  }

  const outputDims = new Array<number>(rank);
  const stride = new Array<number>(rank);

  // determine shape of output, as well as stride to be used
  // stride[i] indicates the stride for the input-tensor dimension
  // corresponding to the i-th dimension of the output
  for (let i = 0; i < rank; i++) {
    const inpDim = finalPerm[i];
    outputDims[i] = inputDims[inpDim];
    if (inpDim + 1 < rank) {
      stride[i] = ShapeUtil.sizeFromDimension(inputDims, inpDim + 1);
    } else {
      stride[i] = 1;
    }
  }

  const output = new Tensor(x.type, TypedArrayUtil.createTypedArray(x.type, ShapeUtil.size(outputDims)), outputDims);

  const X = x.data as NumberDataType;
  const Y = output.data as NumberDataType;

  // partition the permutation into a prefix and the largest suffix such that
  // every axis i in the suffix is mapped to i.
  let numAxesInPrefix = 0;  // number of axes in prefix
  let suffixBlocksize = 1;  // product of dimensions in the suffix
  let prefixBlocksize = 1;  // product of dimensions in the prefix
  let isSuffix = true;
  for (let i = rank - 1; i >= 0; --i) {
    const inpAxis = finalPerm[i];
    if (isSuffix && (inpAxis === i)) {
      suffixBlocksize *= inputDims[inpAxis];
    } else {
      isSuffix = false;
      prefixBlocksize *= inputDims[inpAxis];
      ++numAxesInPrefix;
    }
  }

  if (prefixBlocksize === 1) {
    doTransposeSingleBlock(suffixBlocksize, Y, X);
  } else if (suffixBlocksize === 1) {
    doTransposeEltWise(numAxesInPrefix, outputDims, prefixBlocksize, stride, Y, X);
  } else {
    doTranspose(numAxesInPrefix, outputDims, prefixBlocksize, suffixBlocksize, stride, Y, X);
  }

  return output;
}

// doTranspose: copies source tensor to target, transposing elements.
// the stride vector indicates the transposition.
function doTranspose(
    numAxes: number, targetDims: number[], numBlocks: number, numElementsInBlock: number, stride: number[],
    target: NumberDataType, source: NumberDataType) {
  const targetIndex = new Array<number>(numAxes).fill(0);

  const startSourceIndex = 0;
  let startTargetIndex = 0;

  for (let i = 0; i < numBlocks; ++i) {
    const sizeOffset = ShapeUtil.computeOffset(targetIndex, stride, numAxes);
    arrayCopyHelper(target, source, startTargetIndex, startSourceIndex + sizeOffset, numElementsInBlock);

    ShapeUtil.incrementIndex(targetIndex, targetDims, numAxes);
    startTargetIndex += numElementsInBlock;
  }
}

// doTransposeEltWise: specialization of DoTranspose for the
// num_elts_in_block=1 case. copies source tensor to target, transposing
// elements. The stride vector indicates the transposition.
function doTransposeEltWise(
    numAxes: number, targetDims: number[], numBlocks: number, stride: number[], target: NumberDataType,
    source: NumberDataType) {
  const targetIndex = new Array<number>(numAxes).fill(0);

  let startTargetIndex = 0;

  for (let i = 0; i < numBlocks; ++i) {
    const sourceOffset = ShapeUtil.computeOffset(targetIndex, stride, numAxes);
    target[startTargetIndex++] = source[sourceOffset];
    ShapeUtil.incrementIndex(targetIndex, targetDims, numAxes);
  }
}

// doTransposeSingleBlock: specialization of DoTranspose for the num_blocks=1
// case. copies source tensor to target, transposing elements. The stride
// vector indicates the transposition.
function doTransposeSingleBlock(numElementsInBlock: number, target: NumberDataType, source: NumberDataType) {
  arrayCopyHelper(target, source, 0, 0, numElementsInBlock);
}