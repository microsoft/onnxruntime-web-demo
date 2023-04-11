import _ from "lodash";
import { imagenetClasses } from "../data/imagenet";

export interface ClassResult {
  id: string;
  index: number;
  name: string;
  probability: number;
}

/**
 * Find top k imagenet classes
 */
export function imagenetClassesTopK(
  classProbabilities: any,
  k = 5
): ClassResult[] {
  // console.log("imagenetClassesTopK");
  const probs = _.isTypedArray(classProbabilities)
    ? Array.prototype.slice.call(classProbabilities)
    : classProbabilities;

  const sorted = _.reverse(
    _.sortBy(
      probs.map((prob: any, index: number) => [prob, index]),
      (probIndex: any) => probIndex[0]
    )
  );

  const topK: ClassResult[] = _.take(sorted, k).map((probIndex: any) => {
    const iClass = imagenetClasses[probIndex[1]];
    return {
      id: iClass[0],
      index: parseInt(probIndex[1], 10),
      name: iClass[1].replace(/_/g, " "),
      probability: probIndex[0],
    };
  });
  console.log("topK", topK);
  return topK;
}
