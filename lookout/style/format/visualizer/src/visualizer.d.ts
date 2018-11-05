interface Position {
  offset: number,
  col: number,
  line: number,
}

interface VirtualNode {
  start: Position,
  end: Position,
  value: string,
  path: string,
  roles: number[],
  y: number[],
  internal_type?: string
}

interface RawData {
  code: string,
  features: any,
  ground_truths: number[],
  predictions: number[],
  sibling_indices: number[][],
  rules: string[],
  confidences: number[],
  supports: number[],
  winners: number[],
  feature_names: string[],
  class_representations: string[],
  class_printables: string[],
  vnodes: VirtualNode[],
  config: object,
}

interface AddedData {
  labeled_indices: Map<number, number>,
  rules_by_confidence: number[],
  rule_uls: JSX.Element[],
}

interface Data extends RawData, AddedData {}
