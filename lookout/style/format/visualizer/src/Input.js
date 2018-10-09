import defaultCode from './defaultCode.js'
import React, { Component } from 'react';
import './css/Input.css';

class Input extends Component {

  constructor(props) {
    super(props);
    this.code = React.createRef()
  }

  render() {
    return (
      <div className="Input ">
        <div className="row">
          <div className="col-sm">
            <p>Enter the code to visualize.</p>
          </div>
        </div>
        <div className="row">
          <div className="col-sm">
            <textarea ref={this.code}
                      placeholder="Please enter the code here"
                      defaultValue={defaultCode}>
            </textarea>
          </div>
        </div>
        <div className="row">
          <div className="col-sm">
            <button type="button"
                    className="btn btn-primary btn-lg btn-block"
                    onClick={() => this.props.switchHandler(
                      this.code.current.value)}>
              Visualize
            </button>
          </div>
        </div>
      </div>
    );
  }
}

export default Input;
