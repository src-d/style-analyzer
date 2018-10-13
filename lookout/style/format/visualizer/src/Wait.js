import React, { Component } from "react";
import { Col, Grid, Row } from "react-bootstrap";
import "./css/Wait.css";

class Wait extends Component {
  render() {
    return (
      <Grid fluid={true} className="Wait">
        <Row>
          <Col sm={12}>Pease wait for the server to process your request…</Col>
        </Row>
      </Grid>
    );
  }
}

export default Wait;
