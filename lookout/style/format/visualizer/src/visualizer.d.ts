interface Position {
  offset: number;
  col: number;
  line: number;
}

interface VirtualNode {
  start: Position;
  end: Position;
  value: string;
  path: string;
  y?: number[];
  labeled_index?: number;
}

interface Rule {
  conf: number;
  support: number;
  cls: number;
  attrs: string[];
  artificial: boolean;
}

interface RawData {
  code: string;
  features: any;
  ground_truths: number[];
  predictions: number[];
  sibling_indices: number[][];
  rules: Rule[];
  winners: number[];
  break_uast: boolean[];
  feature_names: string[];
  class_representations: string[];
  class_printables: string[];
  vnodes: VirtualNode[];
  config: object;
}

interface AddedData {
  labeled_indices: Map<number, number>;
  rules_by_confidence: number[];
  rule_uls: JSX.Element[];
}

interface Data extends RawData, AddedData {}
