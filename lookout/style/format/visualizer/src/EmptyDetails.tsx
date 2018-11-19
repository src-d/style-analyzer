import * as React from "react";
import { Col, Grid, Row } from "react-bootstrap";
import "./css/EmptyDetails.css";

const EmptyDetails = () => (
  <Grid fluid={true} className="EmptyDetails">
    <Row>
      <Col sm={12}>No details to display.</Col>
    </Row>
  </Grid>
);

export default EmptyDetails;
