import * as React from "react";
import { Col, Grid, Row } from "react-bootstrap";
import "./css/Wait.css";

const Wait = () => (
  <Grid fluid={true} className="Wait">
    <Row>
      <Col sm={12}>Pease wait for the server to process your request…</Col>
    </Row>
  </Grid>
);

export default Wait;
