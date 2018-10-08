import React, { Component } from 'react';
import Input from './Input.js'
import Visualization from './Visualization.js'
import Wait from './Wait.js'
import './css/App.css';

class App extends Component {

  constructor(props) {
    super(props);

    this.switchToInput = this.switchToInput.bind(this);
    this.switchToVisualization = this.switchToVisualization.bind(this);

    this.state = {
      mode: 'input'
    }
  }

  switchToInput() {
    this.setState({
      mode: 'input',
      response: null
    });
  }

  switchToVisualization(code) {
    this.setState({
      mode: 'wait'
    });
    fetch(this.props.endpoint, {
      'method': 'POST',
      'mode': 'cors',
      'headers': {
        'Content-Type': 'application/json; charset=utf-8',
      },
      'body': JSON.stringify({
        'code': code,
        'babelfish_address': '0.0.0.0:9432',
        'language': 'javascript'
      })
    }).then(results => {
      return results.json();
    }).then(data => {
      data.labeled_indices = []
      let labeled_index = 0;
      data.vnodes.forEach(function(vnode, index) {
        data.labeled_indices.push(vnode.y === null ? null : labeled_index);
        if (vnode.y !== null) {
          labeled_index++;
        }
      });
      this.setState({
        mode: 'visualization',
        data: data
      });
    });
  }

  render() {
    let widget;
    if (this.state.mode === 'input') {
      widget = <Input switchHandler={this.switchToVisualization} />;
    } else if (this.state.mode === 'wait') {
      widget = <Wait />
    } else {
      widget = <Visualization switchHandler={this.switchToInput}
                              data={this.state.data}/>
    }
    return (
      <div className="App">
        {widget}
      </div>
    )
  }
}

export default App;
