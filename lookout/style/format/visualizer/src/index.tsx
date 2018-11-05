import * as React from "react";
import * as ReactDOM from "react-dom";
import App from "./App";
import "./css/index.css";

ReactDOM.render(
  <App endpoint="http://127.0.0.1:5001/" />,
  document.getElementById("root")
);
