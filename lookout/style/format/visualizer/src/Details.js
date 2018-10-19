import React, { Component } from "react";
import { Col, Grid, Row } from "react-bootstrap";
import Inspector from "react-json-inspector";
import "./css/Details.css";

class Details extends Component {
  render() {
    return (
      <Grid fluid={true} className="Details">
        <Row>
          <Col sm={12}>
            <dl className="dl-horizontal">
              <dt>y</dt>
              <dd>
                {this.props.classRepresentations[this.props.y]} ({this.props.y})
              </dd>
              <dt>ŷ</dt>
              <dd>
                {this.props.classRepresentations[this.props.prediction]} (
                {this.props.prediction})
              </dd>
              <dt>Rule</dt>
              <dd>{this.props.rule === null ? "None" : this.props.rule}</dd>
              <dt>Features</dt>
              <dd>
                <Inspector data={this.props.features} search={false} />
              </dd>
            </dl>
          </Col>
        </Row>
      </Grid>
    );
  }
}

export default Details;
