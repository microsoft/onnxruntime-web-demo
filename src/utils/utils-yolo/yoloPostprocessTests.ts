// Important: Run this file in Node.js
// If any utility function fails the test, it will throw an exception

import {Tensor} from 'onnxruntime-web';
import * as tensorTransformUtils from './yoloPostprocess';

scalarTest();
zerosTest();
linspaceTest();
rangeTest();
sigmoidTest();
expTest();
addTest();
subTest();
mulTest();
divTest();
softmaxTest();
concatTest();
stackTest();
gatherTest();
sliceTest();
tileTest();
transposeTest();
expandDimsTest();
greaterEqualTest();
where1DTest();
where2DTest();
castTest();
reshapeTest();
argMax1DTest();
argMax2DTest();
max1DTest();
max2DTest();

console.log('All tests passed!');

function scalarTest() {
  const actual = tensorTransformUtils.scalar(3.14, 'float32');
  const data = new Float32Array(1);
  data[0] = 3.14;
  const expected = new Tensor('float32', data, [1]);
  if (!assertTensorEquality(actual, expected)) {
    throw new Error('Scalar Test failed');
  }
}

function zerosTest() {
  const actual = tensorTransformUtils.zeros([2, 2], 'int32');
  const expected = new Tensor('int32', new Int32Array(4), [2, 2]);
  if (!assertTensorEquality(actual, expected)) {
    throw new Error('Zeros Test failed');
  }
}

function linspaceTest() {
  const actual = tensorTransformUtils.linspace(0, 9, 10);
  const expected = new Tensor('float32', Float32Array.from([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), [10]);
  if (!assertTensorEquality(actual, expected)) {
    throw new Error('Linspace Test failed');
  }
}

function rangeTest() {
  const actual = tensorTransformUtils.range(0, 9, 2);
  const expected = new Tensor('float32', Float32Array.from([0, 2, 4, 6, 8]), [5]);
  if (!assertTensorEquality(actual, expected)) {
    throw new Error('Range Test failed');
  }
}

function sigmoidTest() {
  const actual = tensorTransformUtils.sigmoid(new Tensor('float32', [0, -1, 2, -3]));
  const expected = new Tensor('float32', Float32Array.from([0.5, 0.2689414, 0.8807971, 0.0474259]), [4]);
  if (!assertTensorEquality(actual, expected)) {
    throw new Error('Sigmoid Test failed');
  }
}

function expTest() {
  const actual = tensorTransformUtils.exp(new Tensor('float32', [1, 2, -3]));
  const expected = new Tensor('float32', Float32Array.from([2.7182817, 7.3890562, 0.0497871]), [3]);
  if (!assertTensorEquality(actual, expected)) {
    throw new Error('Exp Test failed');
  }
}

function addTest() {
  const actual = tensorTransformUtils.add(new Tensor('float32', [1, 2, 3]), new Tensor('float32', [10, 20, 30]));
  const expected = new Tensor('float32', Float32Array.from([11, 22, 33]), [3]);
  if (!assertTensorEquality(actual, expected)) {
    throw new Error('Add Test failed');
  }
}

function subTest() {
  const actual = tensorTransformUtils.sub(new Tensor('float32', [1, 2, 3]), new Tensor('float32', [10, 20, 30]));
  const expected = new Tensor('float32', Float32Array.from([-9, -18, -27]), [3]);
  if (!assertTensorEquality(actual, expected)) {
    throw new Error('Sub Test failed');
  }
}

function mulTest() {
  const actual = tensorTransformUtils.mul(new Tensor('float32', [1, 2, 3]), new Tensor('float32', [10, 20, 30]));
  const expected = new Tensor('float32', Float32Array.from([10, 40, 90]), [3]);
  if (!assertTensorEquality(actual, expected)) {
    throw new Error('Mul Test failed');
  }
}

function divTest() {
  const actual = tensorTransformUtils.div(new Tensor('float32', [10, 20, 30]), new Tensor('float32', [1, 2, 3]));
  const expected = new Tensor('float32', Float32Array.from([10, 10, 10]), [3]);
  if (!assertTensorEquality(actual, expected)) {
    throw new Error('Div Test failed');
  }
}

function softmaxTest() {
  const actual = tensorTransformUtils.softmax(new Tensor('float32', [2, 4, 6, 1, 2, 3], [2, 3]));
  const expected = new Tensor(
      'float32', Float32Array.from([0.0158762, 0.1173104, 0.8668135, 0.0900306, 0.2447284, 0.6652408]), [2, 3]);
  if (!assertTensorEquality(actual, expected)) {
    throw new Error('Softmax Test failed');
  }
}

function concatTest() {
  const a = new Tensor('float32', [1, 2]);
  const b = new Tensor('float32', [3, 4]);
  const actual = tensorTransformUtils.concat([a, b]);
  const expected = new Tensor('float32', Float32Array.from([1, 2, 3, 4]), [4]);
  if (!assertTensorEquality(actual, expected)) {
    throw new Error('Concat Test failed');
  }
}

function stackTest() {
  const a = new Tensor('float32', [1, 2]);
  const b = new Tensor('float32', [3, 4]);
  const c = new Tensor('float32', [5, 6]);
  const actual = tensorTransformUtils.stack([a, b, c]);
  const expected = new Tensor('float32', Float32Array.from([1, 2, 3, 4, 5, 6]), [3, 2]);
  if (!assertTensorEquality(actual, expected)) {
    throw new Error('Stack Test failed');
  }
}

function gatherTest() {
  const indices = new Tensor('int32', Int32Array.from([1, 3, 3]));
  const actual = tensorTransformUtils.gather(new Tensor('int32', Int32Array.from([1, 2, 3, 4])), indices);
  const expected = new Tensor('int32', Int32Array.from([2, 4, 4]));
  if (!assertTensorEquality(actual, expected)) {
    throw new Error('Gather Test failed');
  }
}

function sliceTest() {
  const actual = tensorTransformUtils.slice(new Tensor('int32', Int32Array.from([1, 2, 3, 4]), [2, 2]), [1, 0], [1, 2]);
  const expected = new Tensor('int32', Int32Array.from([3, 4]), [1, 2]);
  if (!assertTensorEquality(actual, expected)) {
    throw new Error('Slice Test failed');
  }
}

function tileTest() {
  const actual = tensorTransformUtils.tile(new Tensor('int32', Int32Array.from([1, 2, 3, 4]), [2, 2]), [1, 2]);
  const expected = new Tensor('int32', Int32Array.from([1, 2, 1, 2, 3, 4, 3, 4]), [2, 4]);
  if (!assertTensorEquality(actual, expected)) {
    throw new Error('Tile Test failed');
  }
}

function transposeTest() {
  const actual = tensorTransformUtils.transpose(new Tensor('float32', [1, 2, 3, 4, 5, 6], [2, 3]));
  const expected = new Tensor('float32', Float32Array.from([1, 4, 2, 5, 3, 6]), [3, 2]);
  if (!assertTensorEquality(actual, expected)) {
    throw new Error('Transpose Test failed');
  }
}

function expandDimsTest() {
  const actual = tensorTransformUtils.expandDims(new Tensor('float32', [1, 2, 3, 4]));
  const expected = new Tensor('float32', Float32Array.from([1, 2, 3, 4]), [1, 4]);
  if (!assertTensorEquality(actual, expected)) {
    throw new Error('ExpandDims Test failed');
  }
}

function greaterEqualTest() {
  const actual = tensorTransformUtils.greaterEqual(new Tensor('float32', [1, 2, 3]), new Tensor('float32', [2, 2, 2]));
  const expected = new Tensor('bool', Uint8Array.from([0, 1, 1]), [3]);
  if (!assertTensorEquality(actual, expected)) {
    throw new Error('GreaterEqual Test failed');
  }
}

function where1DTest() {
  const cond = new Tensor('bool', [false, false, true]);
  const a = new Tensor('int32', Int32Array.from([1, 2, 3]));
  const b = new Tensor('int32', Int32Array.from([-1, -2, -3]));
  const actual = tensorTransformUtils.where(cond, a, b);
  const expected = new Tensor('int32', Int32Array.from([-1, -2, 3]), [3]);
  if (!assertTensorEquality(actual, expected)) {
    throw new Error('Where 1D Test failed');
  }
}

function where2DTest() {
  const cond = new Tensor('bool', [false, false, true, true], [2, 2]);
  const a = new Tensor('int32', Int32Array.from([1, 2, 3, 4]), [2, 2]);
  const b = new Tensor('int32', Int32Array.from([-1, -2, -3, -4]), [2, 2]);
  const actual = tensorTransformUtils.where(cond, a, b);
  const expected = new Tensor('int32', Int32Array.from([-1, -2, 3, 4]), [2, 2]);
  if (!assertTensorEquality(actual, expected)) {
    throw new Error('Where 2D Test failed');
  }
}

function castTest() {
  const actual = tensorTransformUtils.cast(new Tensor('float32', [1.5, 2.5, 3]), 'int32');
  const expected = new Tensor('int32', Int32Array.from([1, 2, 3]), [3]);
  if (!assertTensorEquality(actual, expected)) {
    throw new Error('Cast Test failed');
  }
}

function reshapeTest() {
  const actual = tensorTransformUtils.reshape(new Tensor('int32', Int32Array.from([1, 2, 3, 4])), [2, 2]);
  const expected = new Tensor('int32', Int32Array.from([1, 2, 3, 4]), [2, 2]);
  if (!assertTensorEquality(actual, expected)) {
    throw new Error('Reshape Test failed');
  }
}

function argMax1DTest() {
  const actual = tensorTransformUtils.argMax(new Tensor('int32', Int32Array.from([1, 2, 3])));
  const expected = new Tensor('int32', Int32Array.from([2]), [1]);
  if (!assertTensorEquality(actual, expected)) {
    throw new Error('ArgMax 1D Test failed');
  }
}

function argMax2DTest() {
  const actual = tensorTransformUtils.argMax(new Tensor('int32', Int32Array.from([1, 2, 4, 3]), [2, 2]), 1);
  const expected = new Tensor('int32', Int32Array.from([1, 0]), [2]);
  if (!assertTensorEquality(actual, expected)) {
    throw new Error('ArgMax 2D Test failed');
  }
}

function max1DTest() {
  const actual = tensorTransformUtils.max(new Tensor('int32', Int32Array.from([1, 2, 3])));
  const expected = new Tensor('int32', Int32Array.from([3]), [1]);
  if (!assertTensorEquality(actual, expected)) {
    throw new Error('Max 1D Test failed');
  }
}

function max2DTest() {
  const actual = tensorTransformUtils.max(new Tensor('int32', Int32Array.from([1, 2, 3, 4]), [2, 2]), 1);
  const expected = new Tensor('int32', Int32Array.from([2, 4]), [2]);
  if (!assertTensorEquality(actual, expected)) {
    throw new Error('Max 2D Test failed');
  }
}

// Helper for tests
function assertTensorEquality(t1: Tensor, t2: Tensor) {
  // type doesn't match
  if (t1.type !== t2.type) {
    return false;
  }

  // dims don't match
  if (t1.dims.length !== t2.dims.length) {
    return false;
  }

  // dims don't match
  for (let i = 0; i < t1.dims.length; ++i) {
    if (t1.dims[i] !== t2.dims[i]) {
      return false;
    }
  }

  // data doesn't match
  if (t1.data.length !== t2.data.length) {
    return false;
  }

  // data doesn't match
  for (let i = 0; i < t2.data.length; ++i) {
    if (t1.data[i] !== t2.data[i]) {
      if (t1.type === 'string') {
        return false;
      }
      // number type. so allow some precision loss.
      if (Math.abs((t1.data[i] as number) - (t2.data[i] as number)) > 0.0001) {
        return false;
      }
    }
  }

  return true;
}