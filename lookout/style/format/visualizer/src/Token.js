import React, { Component } from "react";
import "./css/Token.css";

class Token extends Component {
  render() {
    const props = this.props;
    let span = null;
    if (props.y !== null) {
      span = (
        <span
          key={props.index}
          className={
            "feature" +
            (props.highlighted ? " highlighted" : "") +
            (props.neighbour ? " neighbour" : "") +
            (props.enabled && props.correct ? " correct" : "") +
            (props.enabled && !props.correct ? " wrong" : "") +
            (!props.enabled ? " disabled" : "")
          }
          onClick={() => props.highlightCallback(props.index)}
        >
          {props.classPrintables[props.y]}
        </span>
      );
    } else {
      span = (
        <span className={"context" + (props.neighbour ? " neighbour" : "")}>
          {props.value}
        </span>
      );
    }
    return span;
  }
}

export default Token;
