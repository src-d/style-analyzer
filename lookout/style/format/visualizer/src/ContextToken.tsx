// @flow
import * as React from "react";
import "./css/Token.css";

export interface IProps {
  neighbour: boolean;
  value: string;
}

const ContextToken = (props: IProps) => {
  return (
    <span className={"context" + (props.neighbour ? " neighbour" : "")}>
      {props.value}
    </span>
  );
};

export default ContextToken;
