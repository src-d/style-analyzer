import defaultCode from "./defaultCode.js";
import React, { Component } from "react";
import { Col, Grid, Row } from "react-bootstrap";
import "./css/Input.css";

class Input extends Component {
  constructor(props) {
    super(props);
    this.code = React.createRef();
  }

  render() {
    return (
      <Grid fluid={true} className="Input">
        <Row>
          <Col sm={12}>
            <p>Enter the code to visualize.</p>
          </Col>
        </Row>
        <Row>
          <Col sm={12}>
            <textarea
              ref={this.code}
              placeholder="Please enter the code here"
              defaultValue={defaultCode}
            />
          </Col>
        </Row>
        <Row>
          <Col sm={12}>
            <button
              type="button"
              className="btn btn-primary btn-lg btn-block"
              onClick={() => this.props.switchHandler(this.code.current.value)}
            >
              Visualize
            </button>
          </Col>
        </Row>
      </Grid>
    );
  }
}

export default Input;
