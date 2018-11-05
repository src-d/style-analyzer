import * as React from "react";
import "./css/App.css";
import Input from "./Input";
import Visualization from "./Visualization";
import Wait from "./Wait";

export interface IProps {
  endpoint: string,
}

export interface IState {
  data?: Data,
  mode: string,
}

class App extends React.Component<IProps, IState> {
  constructor(props: IProps) {
    super(props);

    this.state = {
      mode: "input"
    };
  }

  public switchToInput = (): void => {
    this.setState({
      data: undefined,
      mode: "input",
    });
  }

  public switchToVisualization = (code: string) => {
    this.setState({
      mode: "wait"
    });
    fetch(this.props.endpoint, {
      body: JSON.stringify({
        babelfish_address: "0.0.0.0:9432",
        code,
        language: "javascript",
      }),
      headers: {
        "Content-Type": "application/json; charset=utf-8",
      },
      method: "POST",
      mode: "cors",
    })
      .then(results => {
        return results.json();
      })
      .then((rawData: Data) => {
        const data: Data = rawData;
        let labeledIndex = 0;
        data.labeled_indices = new Map();
        data.vnodes.forEach((vnode, index) => {
          if (vnode.y !== null) {
            data.labeled_indices.set(index, labeledIndex);
            labeledIndex++;
          }
        });
        const rulesByConfidence = data.confidences.map((confidence, index) => [index, confidence]);
        rulesByConfidence.sort(
          ([_indexLeft, confLeft], [_indexRight, confRight]) =>
            confLeft > confRight ? 1 : confLeft < confRight ? -1 : 0
        );
        data.rules_by_confidence = rulesByConfidence.map(
          ([index, _conf]) => index
        );
        data.rule_uls = data.rules.map((rule, indexRule) => {
          const parts = rule.split("\n\t").map((part, indexPart) =>
            <li key={indexRule * 1000 + indexPart}>{part}</li>);
          return <ul key={indexRule} className="list-unstyled">{parts}</ul>;
        });
        this.setState({
          data,
          mode: "visualization",
        });
      });
  }

  public render() {
    let widget;
    if (this.state.mode === "input") {
      widget = <Input switchHandler={this.switchToVisualization} />;
    } else if (this.state.mode === "wait") {
      widget = <Wait />;
    } else if (this.state.mode === "visualization" && this.state.data !== undefined){
      widget = (
        <Visualization
          switchHandler={this.switchToInput}
          data={this.state.data}
        />
      );
    }
    return <div className="App">{widget}</div>;
  }
}

export default App;
