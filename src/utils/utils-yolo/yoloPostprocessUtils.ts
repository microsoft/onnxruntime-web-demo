import ndarray from 'ndarray';

import {NumberDataType, Type} from './yoloPostprocess';

export class BroadcastUtil {
  /**
   * Calculate the expected shape when broadcasting 2 tensors
   * @param a The shape of tensor A. Should be an array of positive integers
   * @param b The shape of tensor B. Should be an array of positive integers
   * @param isMatMul Whether the operation is MatMul
   * @returns The expected shape of the result, or undefined if N/A
   */
  static calcShape(adims: ReadonlyArray<number>, bdims: ReadonlyArray<number>, isMatMul = false): number[]|undefined {
    const arank = adims.length;
    const brank = bdims.length;
    const crank = Math.max(adims.length, bdims.length);
    const cdims = new Array<number>(crank);

    // calculate the last 2 dimension if it is MatMul
    if (isMatMul) {
      if (arank < 2 || brank < 2) {
        return undefined;
      }
      const cShapeMatMul =
          BroadcastUtil.calcMatMulShape([adims[arank - 2], adims[arank - 1]], [bdims[brank - 2], bdims[brank - 1]]);
      if (cShapeMatMul === undefined) {
        return undefined;
      }
      [cdims[crank - 2], cdims[crank - 1]] = cShapeMatMul;
    }

    for (let i = isMatMul ? 3 : 1; i <= crank; i++) {
      const aLen = arank - i < 0 ? 1 : adims[arank - i];
      const bLen = brank - i < 0 ? 1 : bdims[brank - i];

      if (aLen !== bLen && aLen > 1 && bLen > 1) {
        return undefined;
      }
      cdims[crank - i] = Math.max(aLen, bLen);
    }

    return cdims;
  }

  /**
   * Calculate the expected shape when matrix multiplication
   * @param a The shape of tensor A. Should be a tuple of 2 positive integers
   * @param b The shape of tensor B. Should be a tuple of 2 positive integers
   * @returns The expected shape of the result, or undefined if N/A
   */
  static calcMatMulShape(a: [number, number], b: [number, number]): [number, number]|undefined {
    return (a[1] !== b[0]) ? undefined : [a[0], b[1]];
  }

  /**
   * Given the indices of a broadcasted tensor, calculate the original indices
   * @param indices The given indices of the broadcasted tensor.
   * @param shapeOrigin The origin shape of the tensor before broadcast
   * @param isMatMul Whether the operation is MatMul
   * @returns The calculated indices that maps to the original tensor. If the
   * operation is MatMul, the indices of last 2 dimensions will keep as same as
   * input indices
   */
  static index(indices: number[], shapeOrigin: number[], isMatMul = false): number[] {
    // we assume the parameter indices is valid. ie. it should have the same
    // length as the broadcasted shape, and for each dimension the index should
    // not be out of range.
    const dimOffset = indices.length - shapeOrigin.length;
    const indicesOrigin = indices.slice(dimOffset);
    const dimLen = isMatMul ? indicesOrigin.length - 2 : indicesOrigin.length;
    for (let i = 0; i < dimLen; i++) {
      indicesOrigin[i] = indices[dimOffset + i] % shapeOrigin[i];
    }
    return indicesOrigin;
  }

  /**
   * Perform the broadcasting operation on the specific operator
   * @param a The input tensor A
   * @param b The input tensor B
   * @param op The operator lambda function
   * @returns The result tensor, or undefined if input not broadcastable.
   */
  static calc(a: ndarray.NdArray, b: ndarray.NdArray, op: (a: number, b: number) => number): ndarray.NdArray|undefined {
    const shape = BroadcastUtil.calcShape(a.shape, b.shape);
    if (shape) {
      const size = ShapeUtil.size(shape);
      const c = ndarray(
          new (
              a.data.constructor as Int8ArrayConstructor | Int16ArrayConstructor | Int32ArrayConstructor |
              Uint8ArrayConstructor | Uint16ArrayConstructor | Uint32ArrayConstructor | Float32ArrayConstructor |
              Float64ArrayConstructor | Uint8ClampedArrayConstructor)(size),
          shape);

      const indices = new Array<number>(shape.length);
      for (let i = 0; i < size; i++) {
        // traversal indices
        let rest = i;
        for (let j = shape.length - 1; j >= 0; j--) {
          indices[j] = rest % shape[j];
          rest = Math.floor(rest / shape[j]);
        }

        // map index
        const indicesA = BroadcastUtil.index(indices, a.shape);
        const indicesB = BroadcastUtil.index(indices, b.shape);

        // assign value
        c.set(...indices.concat(op(a.get(...indicesA), b.get(...indicesB))));
      }

      return c;
    }

    return undefined;
  }

  /**
   * Determine if a shape is unidirectional broadcastable to another shape
   * @param shape The input shape
   * @param finalShape The desired shape after broadcasting
   */
  static isValidBroadcast(shape: ReadonlyArray<number>, finalShape: ReadonlyArray<number>): boolean {
    // align shape to the right
    const inputRank = shape.length;
    const finalRank = finalShape.length;
    if (inputRank > finalRank) {
      return false;
    }
    for (let i = 1; i <= inputRank; i++) {
      if (shape[inputRank - i] !== 1 && shape[inputRank - i] !== finalShape[finalRank - i]) {
        return false;
      }
    }
    return true;
  }
}
// copy array helper
// mimics memcpy as much as possible
export function arrayCopyHelper(
    target: NumberDataType, source: NumberDataType, targetIndex: number, sourceIndex: number, blockSize: number) {
  if (sourceIndex < 0 || sourceIndex >= source.length) {
    throw new Error(`sourceIndex out of bounds`);
  }
  if (targetIndex < 0 || targetIndex >= target.length) {
    throw new Error(`targetIndex out of bounds`);
  }
  if (sourceIndex + blockSize > source.length) {
    throw new Error(`source indices to be copied are outside bounds`);
  }
  if (targetIndex + blockSize > target.length) {
    throw new Error(`target array is too small to hold result`);
  }

  for (let offset = 0; offset < blockSize; offset++) {
    target[targetIndex + offset] = source[sourceIndex + offset];
  }
}

export class TypeUtil {
  static validateSameTypes(typesArray: Type[]) {
    if (typesArray.length < 2) {
      throw new Error('must contain atleast 2 types to compare equality');
    }
    const baseType = typesArray[0];
    for (let i = 0; i < typesArray.length; ++i) {
      if (typesArray[i] !== baseType) {
        throw new Error('input types are ');
      }
    }
  }
}

export class ShapeUtil {
  static validateEqualDims(dimsArray: Array<ReadonlyArray<number>>) {
    if (dimsArray.length < 2) {
      throw new Error('must contain atleast 2 shapes to compare equality');
    }
    const baseDims = dimsArray[0];
    const baseRank = baseDims.length;
    for (let i = 1; i < dimsArray.length; ++i) {
      const dims = dimsArray[i];
      if (dims.length !== baseRank) {
        throw new Error('rank is not the same for given inpu shapes');
      }
      for (let j = 0; j < baseRank; ++j) {
        if (baseDims[j] !== dims[j]) {
          throw new Error('input shapes are not the same');
        }
      }
    }
  }

  static validateDims(dims: ReadonlyArray<number>) {
    if (dims.length < 0 || dims.length > 6) {
      throw new TypeError(`Only rank 0 to 6 is supported for tensor shape.`);
    }

    if (dims.length === 0) {
      throw new RangeError('Scaler tensor is not implemented yet');
    }

    for (const n of dims) {
      if (!Number.isInteger(n)) {
        throw new TypeError(`Invalid shape: ${n} is not an integer`);
      }
      if (n <= 0 || n > 2147483647) {
        throw new TypeError(`Invalid shape: length ${n} is not allowed`);
      }
    }
  }

  static size(dims: ReadonlyArray<number>): number {
    return ShapeUtil.getSizeFromDimensionRange(dims, 0, dims.length);
  }

  static sizeFromDimension(dims: ReadonlyArray<number>, axis: number): number {
    if (axis > dims.length) {
      throw new Error(`invalid dimension of ${axis} for sizeFromDimension as Tensor has ${dims.length} dimensions.`);
    }

    return ShapeUtil.getSizeFromDimensionRange(dims, axis, dims.length);
  }

  static sizeToDimension(dims: ReadonlyArray<number>, axis: number): number {
    if (axis > dims.length) {
      throw new Error(`invalid dimension of ${axis} for sizeToDimension as Tensor has ${dims.length} dimensions.`);
    }

    return ShapeUtil.getSizeFromDimensionRange(dims, 0, axis);
  }

  static getSizeFromDimensionRange(dims: ReadonlyArray<number>, start: number, end: number): number {
    let size = 1;
    for (let i = start; i < end; i++) {
      // safety check as this method is called by multiple other methods requiring size.
      // size cannot be 0 or negative.
      if (dims[i] <= 0) {
        throw new Error(
            // tslint:disable-next-line:max-line-length
            `cannot get valid size from specified dimension range. Most likely the range contains 0 or negative values in them.`);
      }
      size *= dims[i];
    }
    return size;
  }

  // Computes the offset up until the start index for the specified axis
  /**
   * @param index Given index to compute offset for in the flattened
   * @param stride The strides of the tensor corresponding to the index
   * @param axis The 1-indexed axis upto which the offset is to be computed for. If undefined, axis == rank of the
   * index.
   */

  static computeOffset(index: number[], stride: number[], axis?: number) {
    if (axis === undefined) {
      axis = index.length;
    }
    let offset = 0;
    for (let i = 0; i < axis; ++i) {
      offset += (index[i] * stride[i]);
    }
    return offset;
  }
  static computeStrides(shape: ReadonlyArray<number>): number[] {
    const rank = shape.length;
    if (rank < 2) {
      return [1];
    }

    const strides = new Array(rank);
    strides[rank - 1] = 1;
    strides[rank - 2] = shape[rank - 1];
    for (let i = rank - 3; i >= 0; --i) {
      strides[i] = strides[i + 1] * shape[i + 1];
    }
    return strides;
  }
  static transpose(dims: number[]): number[] {
    return dims.reverse();
  }
  static indicesToOffset(indices: number[], strides: number[]): number {
    const rank = strides.length;
    if (rank === 0) {
      return 0;
    }
    let index = indices[indices.length - 1];
    for (let i = 0; i < indices.length - 1; ++i) {
      index += strides[i] * indices[i];
    }
    return index;
  }

  static offsetToIndices(offset: number, strides: number[]): number[] {
    const rank = strides.length;
    if (rank === 0) {
      return [];
    } else if (rank === 1) {
      return [offset];
    }
    const indices: number[] = new Array(strides.length);
    for (let i = 0; i < indices.length - 1; ++i) {
      indices[i] = Math.floor(offset / strides[i]);
      offset -= indices[i] * strides[i];
    }
    indices[indices.length - 1] = offset;
    return indices;
  }
  static getActualAxisFromNegativeValue(axis: number, tensorRank: number): number {
    if (axis < -tensorRank && axis >= tensorRank - 1) {
      throw new Error('unsupported axis for this operation.');
    }
    return axis < 0 ? axis + tensorRank : axis;
  }

  // Increment an index into a tensor (in lexicographic
  // ordering), wrapping around the specified upper_bound.
  /**
   * Increment an index into a tensor (in lexicographic ordering), wrapping around the specified upper_bound.
   * @param index Given index to increment
   * @param dims The dimensions of the tensor for which the given index corresponds to
   * @param axisToIncrementOn The 1-indexed axis to increment on. If undefined, axisToIncrementOn == rank
   */
  static incrementIndex(index: number[], dims: number[], axisToIncrementOn?: number) {
    if (axisToIncrementOn === undefined) {
      axisToIncrementOn = dims.length;
    }

    for (let k = axisToIncrementOn - 1; k >= 0; --k) {
      index[k]++;
      if (index[k] < dims[k]) {
        break;
      }
      index[k] = 0;
    }
  }

  /**
   * Produces a new dimensions array based on the values in the 'originalDimensions' and 'shape' array
   * Used in Reshape
   * @param originalDims Original Shape array
   * @param shapeHints array containing values to compute the new dimensions
   * For example:
   * originalDims = [2,2] and shapeHints = [0,-1] will return [2,2]
   * originalDims = [2,2] and shapeHints = [4] will return [4]
   * originalDims = [2,2] and shapeHints = [5] will throw an exception
   * https://github.com/onnx/onnx/blob/master/docs/Operators.md#Reshape
   */

  static calculateReshapedDims(originalDims: ReadonlyArray<number>, shapeHints: ReadonlyArray<number>): number[] {
    const nDims = shapeHints.length;
    const reshapedDims = new Array<number>(nDims);
    let unknownDimension = -1;
    let size = 1;

    for (let i = 0; i < nDims; i++) {
      if (shapeHints[i] < -1) {
        throw new Error('a dimension cannot be less than -1');
      }
      if (shapeHints[i] === -1) {
        if (unknownDimension !== -1) {
          throw new Error('at most one dimension can be -1');
        }
        unknownDimension = i;
      } else {
        if (shapeHints[i] === 0) {
          if (i >= originalDims.length) {
            throw new Error('the dimension with value zero exceeds the dimension size of the input tensor');
          }
          reshapedDims[i] = originalDims[i];
        } else {
          reshapedDims[i] = shapeHints[i];
        }
        size *= reshapedDims[i];
      }
    }

    if (unknownDimension !== -1) {
      const originalTensorFlattenedSize = ShapeUtil.size(originalDims);
      if (originalTensorFlattenedSize % size !== 0) {
        throw new Error(`the input tensor cannot be reshaped to the requested shape. Input shape: [${
            originalDims}] Output shape: [${shapeHints}]`);
      }
      reshapedDims[unknownDimension] = originalTensorFlattenedSize / size;
    }
    return reshapedDims;
  }

  /**
   * Sorts a given array based on the indices in the Perm array
   * Used in Transpose
   * @param a Array to be sorted such as dims or strides
   * @param perm Perm given; if null a will be reversed
   */
  static sortBasedOnPerm(a: ReadonlyArray<number>, perm?: number[]): number[] {
    if (perm) {
      return perm.map((v) => a[v]);
    } else {
      return a.slice().reverse();
    }
  }

  /**
   * Pads a given shape according to the padding values
   * @param dims shape of the Tensor to be padded
   * @param pad pad values
   */
  static padShape(dims: ReadonlyArray<number>, pad: number[]): number[] {
    const rank = dims.length;
    return dims.map((v, i) => v + pad[i] + pad[i + rank]);
  }

  /**
   * Determines if the two shapes are identical
   * @param shape1
   * @param shape2
   */
  static areEqual(shape1: ReadonlyArray<number>, shape2: ReadonlyArray<number>): boolean {
    if (shape1.length !== shape2.length) {
      return false;
    }
    return shape1.every((v, i) => v === shape2[i]);
  }

  /**
   * Splits a given `dims` into 2 mutually exclusive `dims`
   * @param dims ReadonlyArray<number>
   * @param pick number - picks the dim along this axis and composes a new `dims`.
   * The remnants make up another `dims`
   */
  static splitDimsIntoTwo(dims: ReadonlyArray<number>, pick: number): [number[], number[]] {
    const picked: number[] = [];
    const remnants: number[] = [];

    for (let i = 0; i < dims.length; ++i) {
      if (i === pick) {
        picked.push(dims[i]);
      } else {
        remnants.push(dims[i]);
      }
    }

    return [picked, remnants];
  }
}

export class TypedArrayUtil {
  static createTypedArray(type: string, size: number): Uint8Array|Int32Array|Float32Array {
    switch (type) {
      case 'bool':
        return new Uint8Array(size);
      case 'int32':
        return new Int32Array(size);
      case 'float32':
        return new Float32Array(size);
      default:
        throw new Error('Unsupported type');
    }
  }
}