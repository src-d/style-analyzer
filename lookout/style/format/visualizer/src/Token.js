import React, { Component } from 'react';
import './css/Token.css';

class Token extends Component {

  render() {
    let span = null;
    if (this.props.y !== null) {
      span = <span key={this.props.index}
                   className={"feature" + (this.props.highlighted ? " highlighted" : "")
                                        + (this.props.neighbour ? " neighbour" : "")
                                        + (this.props.correct ? " correct" : " wrong")}
                   onClick={() => this.props.highlightCallback(this.props.index)}>{
        this.props.classPrintables[this.props.y]
      }</span>
    } else {
      span = <span className={"context" + (this.props.neighbour ? " neighbour" : "")}
             >{this.props.value}</span>
    }
    return span;
  }
}

export default Token;
