declare module 'blueimp-load-image' {
  export default function loadImage(
      url: File|Blob|string, callback: (img: HTMLImageElement|HTMLCanvasElement|Event) => void, {}): any;
}

declare module 'ndarray-gemm' {
import {NdArray} from 'NdArray';
  export default function matrixProduct(c: NdArray, a: NdArray, b: NdArray, alpha?: number, beta?: number): void;
}

declare module 'ndarray-ops' {
import {NdArray, Data} from 'NdArray';
  export function assign(dest: NdArray, src: NdArray): NdArray;
  export function assigns(dest: NdArray, val: Data): NdArray;

  // add[,s,eq,seq] - Addition, +
  export function add(dest: NdArray, arg1: NdArray, arg2: NdArray): NdArray;
  export function adds(dest: NdArray, arg1: NdArray, scalar: number): NdArray;
  export function addeq(dest: NdArray, arg1: NdArray): NdArray;
  export function addseq(dest: NdArray, scalar: number): NdArray;

  // sub[,s,eq,seq] - Subtraction, -
  export function sub(dest: NdArray, arg1: NdArray, arg2: NdArray): NdArray;
  export function subs(dest: NdArray, arg1: NdArray, scalar: number): NdArray;
  export function subeq(dest: NdArray, arg1: NdArray): NdArray;
  export function subseq(dest: NdArray, scalar: number): NdArray;

  // mul[,s,eq,seq] - Multiplication, *
  export function mul(dest: NdArray, arg1: NdArray, arg2: NdArray): NdArray;
  export function muls(dest: NdArray, arg1: NdArray, scalar: number): NdArray;
  export function muleq(dest: NdArray, arg1: NdArray): NdArray;
  export function mulseq(dest: NdArray, scalar: number): NdArray;

  // div[,s,eq,seq] - Division, /
  export function div(dest: NdArray, arg1: NdArray, arg2: NdArray): NdArray;
  export function divs(dest: NdArray, arg1: NdArray, scalar: number): NdArray;
  export function diveq(dest: NdArray, arg1: NdArray): NdArray;
  export function divseq(dest: NdArray, scalar: number): NdArray;

  // mod[,s,eq,seq] - Modulo, %
  export function mod(dest: NdArray, arg1: NdArray, arg2: NdArray): NdArray;
  export function mods(dest: NdArray, arg1: NdArray, scalar: number): NdArray;
  export function modeq(dest: NdArray, arg1: NdArray): NdArray;
  export function modseq(dest: NdArray, scalar: number): NdArray;

  // band[,s,eq,seq] - Bitwise And, &
  export function band(dest: NdArray, arg1: NdArray, arg2: NdArray): NdArray;
  export function bands(dest: NdArray, arg1: NdArray, scalar: number): NdArray;
  export function bandeq(dest: NdArray, arg1: NdArray): NdArray;
  export function bandseq(dest: NdArray, scalar: number): NdArray;

  // bor[,s,eq,seq] - Bitwise Or, &
  export function bor(dest: NdArray, arg1: NdArray, arg2: NdArray): NdArray;
  export function bors(dest: NdArray, arg1: NdArray, scalar: number): NdArray;
  export function boreq(dest: NdArray, arg1: NdArray): NdArray;
  export function borseq(dest: NdArray, scalar: number): NdArray;

  // bxor[,s,eq,seq] - Bitwise Xor, ^
  export function bxor(dest: NdArray, arg1: NdArray, arg2: NdArray): NdArray;
  export function bxors(dest: NdArray, arg1: NdArray, scalar: number): NdArray;
  export function bxoreq(dest: NdArray, arg1: NdArray): NdArray;
  export function bxorseq(dest: NdArray, scalar: number): NdArray;

  // lshift[,s,eq,seq] - Left shift, <<
  export function lshift(dest: NdArray, arg1: NdArray, arg2: NdArray): NdArray;
  export function lshifts(dest: NdArray, arg1: NdArray, scalar: number): NdArray;
  export function lshifteq(dest: NdArray, arg1: NdArray): NdArray;
  export function lshiftseq(dest: NdArray, scalar: number): NdArray;

  // rshift[,s,eq,seq] - Signed right shift, >>
  export function rshift(dest: NdArray, arg1: NdArray, arg2: NdArray): NdArray;
  export function rshifts(dest: NdArray, arg1: NdArray, scalar: number): NdArray;
  export function rshifteq(dest: NdArray, arg1: NdArray): NdArray;
  export function rshiftseq(dest: NdArray, scalar: number): NdArray;

  // rrshift[,s,eq,seq] - Unsigned right shift, >>>
  export function rrshift(dest: NdArray, arg1: NdArray, arg2: NdArray): NdArray;
  export function rrshifts(dest: NdArray, arg1: NdArray, scalar: number): NdArray;
  export function rrshifteq(dest: NdArray, arg1: NdArray): NdArray;
  export function rrshiftseq(dest: NdArray, scalar: number): NdArray;

  // lt[,s,eq,seq] - Less than, <
  export function lt(dest: NdArray, arg1: NdArray, arg2: NdArray): NdArray;
  export function lts(dest: NdArray, arg1: NdArray, scalar: number): NdArray;
  export function lteq(dest: NdArray, arg1: NdArray): NdArray;
  export function ltseq(dest: NdArray, scalar: number): NdArray;

  // gt[,s,eq,seq] - Greater than, >
  export function gt(dest: NdArray, arg1: NdArray, arg2: NdArray): NdArray;
  export function gts(dest: NdArray, arg1: NdArray, scalar: number): NdArray;
  export function gteq(dest: NdArray, arg1: NdArray): NdArray;
  export function gtseq(dest: NdArray, scalar: number): NdArray;

  // leq[,s,eq,seq] - Less than or equal, <=
  export function leq(dest: NdArray, arg1: NdArray, arg2: NdArray): NdArray;
  export function leqs(dest: NdArray, arg1: NdArray, scalar: number): NdArray;
  export function leqeq(dest: NdArray, arg1: NdArray): NdArray;
  export function leqseq(dest: NdArray, scalar: number): NdArray;

  // geq[,s,eq,seq] - Greater than or equal >=
  export function geq(dest: NdArray, arg1: NdArray, arg2: NdArray): NdArray;
  export function geqs(dest: NdArray, arg1: NdArray, scalar: number): NdArray;
  export function geqeq(dest: NdArray, arg1: NdArray): NdArray;
  export function geqseq(dest: NdArray, scalar: number): NdArray;

  // eq[,s,eq,seq] - Equals, ===
  export function eq(dest: NdArray, arg1: NdArray, arg2: NdArray): NdArray;
  export function eqs(dest: NdArray, arg1: NdArray, scalar: number): NdArray;
  export function eqeq(dest: NdArray, arg1: NdArray): NdArray;
  export function eqseq(dest: NdArray, scalar: number): NdArray;

  // neq[,s,eq,seq] - Not equals, !==
  export function neq(dest: NdArray, arg1: NdArray, arg2: NdArray): NdArray;
  export function neqs(dest: NdArray, arg1: NdArray, scalar: number): NdArray;
  export function neqeq(dest: NdArray, arg1: NdArray): NdArray;
  export function neqseq(dest: NdArray, scalar: number): NdArray;

  // and[,s,eq,seq] - Boolean And, &&
  export function and(dest: NdArray, arg1: NdArray, arg2: NdArray): NdArray;
  export function ands(dest: NdArray, arg1: NdArray, scalar: number): NdArray;
  export function andeq(dest: NdArray, arg1: NdArray): NdArray;
  export function andseq(dest: NdArray, scalar: number): NdArray;

  // or[,s,eq,seq] - Boolean Or, ||
  export function or(dest: NdArray, arg1: NdArray, arg2: NdArray): NdArray;
  export function ors(dest: NdArray, arg1: NdArray, scalar: number): NdArray;
  export function oreq(dest: NdArray, arg1: NdArray): NdArray;
  export function orseq(dest: NdArray, scalar: number): NdArray;

  // max[,s,eq,seq] - Maximum, Math.max
  export function max(dest: NdArray, arg1: NdArray, arg2: NdArray): NdArray;
  export function maxs(dest: NdArray, arg1: NdArray, scalar: number): NdArray;
  export function maxeq(dest: NdArray, arg1: NdArray): NdArray;
  export function maxseq(dest: NdArray, scalar: number): NdArray;

  // min[,s,eq,seq] - Minimum, Math.min
  export function min(dest: NdArray, arg1: NdArray, arg2: NdArray): NdArray;
  export function mins(dest: NdArray, arg1: NdArray, scalar: number): NdArray;
  export function mineq(dest: NdArray, arg1: NdArray): NdArray;
  export function minseq(dest: NdArray, scalar: number): NdArray;

  // not[,eq] - Boolean not, !
  export function not(dest: NdArray, arg: NdArray): NdArray;
  export function noteq(dest: NdArray): NdArray;

  // bnot[,eq] - Bitwise not, ~
  export function bnot(dest: NdArray, arg: NdArray): NdArray;
  export function bnoteq(dest: NdArray): NdArray;

  // neg[,eq] - Negative, -
  export function neg(dest: NdArray, arg: NdArray): NdArray;
  export function negeq(dest: NdArray): NdArray;

  // recip[,eq] - Reciprocal, 1.0/
  export function recip(dest: NdArray, arg: NdArray): NdArray;
  export function recipeq(dest: NdArray): NdArray;

  // abs[,eq] - Absolute value, Math.abs
  export function abs(dest: NdArray, arg: NdArray): NdArray;
  export function abseq(dest: NdArray): NdArray;

  // acos[,eq] - Inverse cosine, Math.acos
  export function acos(dest: NdArray, arg: NdArray): NdArray;
  export function acoseq(dest: NdArray): NdArray;

  // asin[,eq] - Inverse sine, Math.asin
  export function asin(dest: NdArray, arg: NdArray): NdArray;
  export function asineq(dest: NdArray): NdArray;

  // atan[,eq] - Inverse tangent, Math.atan
  export function atan(dest: NdArray, arg: NdArray): NdArray;
  export function ataneq(dest: NdArray): NdArray;

  // ceil[,eq] - Ceiling, Math.ceil
  export function ceil(dest: NdArray, arg: NdArray): NdArray;
  export function ceileq(dest: NdArray): NdArray;

  // cos[,eq] - Cosine, Math.cos
  export function cos(dest: NdArray, arg: NdArray): NdArray;
  export function coseq(dest: NdArray): NdArray;

  // exp[,eq] - Exponent, Math.exp
  export function exp(dest: NdArray, arg: NdArray): NdArray;
  export function expeq(dest: NdArray): NdArray;

  // floor[,eq] - Floor, Math.floor
  export function floor(dest: NdArray, arg: NdArray): NdArray;
  export function flooreq(dest: NdArray): NdArray;

  // log[,eq] - Logarithm, Math.log
  export function log(dest: NdArray, arg: NdArray): NdArray;
  export function logeq(dest: NdArray): NdArray;

  // round[,eq] - Round, Math.round
  export function round(dest: NdArray, arg: NdArray): NdArray;
  export function roundeq(dest: NdArray): NdArray;

  // sin[,eq] - Sine, Math.sin
  export function sin(dest: NdArray, arg: NdArray): NdArray;
  export function sineq(dest: NdArray): NdArray;

  // sqrt[,eq] - Square root, Math.sqrt
  export function sqrt(dest: NdArray, arg: NdArray): NdArray;
  export function sqrteq(dest: NdArray): NdArray;

  // tan[,eq] - Tangent, Math.tan
  export function tan(dest: NdArray, arg: NdArray): NdArray;
  export function taneq(dest: NdArray): NdArray;
}
