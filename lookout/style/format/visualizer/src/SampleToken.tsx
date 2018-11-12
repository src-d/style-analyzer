// @flow
import * as React from "react";
import "./css/Token.css";

export interface IProps {
  correct: boolean;
  enabled: boolean;
  highlightCallback: (index: number) => void;
  highlighted: boolean;
  index: number;
  neighbour: boolean;
  value: string;
}

const SampleToken = (props: IProps) => {
  return (
    <span
      className={
        "feature" +
        (props.highlighted ? " highlighted" : "") +
        (props.neighbour ? " neighbour" : "") +
        (props.enabled && props.correct ? " correct" : "") +
        (props.enabled && !props.correct ? " wrong" : "") +
        (!props.enabled ? " disabled" : "")
      }
      onClick={_e => props.highlightCallback(props.index)}
    >
      {props.value}
    </span>
  );
};

export default SampleToken;
