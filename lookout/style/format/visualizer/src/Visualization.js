import Features from './Features.js'
import React, { Component } from 'react';
import Token from './Token.js'
import './css/Visualization.css';

class Visualization extends Component {

  constructor(props) {
    super(props);
    this.findSiblings = this.findSiblings.bind(this);
    this.classify = this.classify.bind(this);
    this.highlight = this.highlight.bind(this);
    this.state = {
      highlighted: null,
      neighbours: []
    };
  }

  onClick(e, text) {
    console.log(e.target, text);
  }

  findSiblings(key) {
    return this.props.data.sibling_indices[
      this.props.data.labeled_indices[key]];
  }

  classify(key) {
    if (this.props.data.labeled_indices[key] !== undefined) {
      return this.props.data.vnodes[key].y === this.props.data.predictions[
        this.props.data.labeled_indices[key]];
    } else {
      return false;
    }
  }

  highlight(key) {
    if (key === this.state.highlighted) {
      this.setState({highlighted: null,
                     neighbours: []})
    } else {
      this.setState({highlighted: key,
                     neighbours: this.findSiblings(key)})
    }
  }

  render() {
    const data = this.props.data;
    const printables = data.class_printables;
    const tokens = [];
    data.vnodes.forEach((vnode, index) => {
      tokens.push(<Token key={index.toString()}
                         index={index}
                         highlighted={this.state.highlighted === index}
                         neighbour={this.state.neighbours.includes(index)}
                         correct={this.classify(index)}
                         value={vnode.value}
                         y={vnode.y}
                         classPrintables={printables}
                         highlightCallback={this.highlight} />);
    });
    return (
      <div className="Visualization">
        <div className="row">
          <div className="col-sm">
            <p>Click on a token to display its features.</p>
          </div>
        </div>
        <div className="row">
          <div className="col-sm-9">
            <div className="code-display">
              <pre>
                <code>
                  {tokens}
                </code>
              </pre>
            </div>
          </div>
          <div className="col-sm-3">
            <Features features={this.state.highlighted !== null
                                ? this.props.data.vnodes[this.state.highlighted]
                                : null}
                      classRepresentations={this.props.data.class_representations} />
          </div>
        </div>
        <div className="row">
          <div className="col-sm">
            <button type="button"
                    className="btn btn-primary btn-lg btn-block"
                    onClick={this.props.switchHandler}>
              Try another input
            </button>
          </div>
        </div>
      </div>
    );
  }
}

export default Visualization;
