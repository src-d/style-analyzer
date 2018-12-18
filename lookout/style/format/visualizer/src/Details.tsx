import * as React from "react";
import { Col, Grid, Row } from "react-bootstrap";
import Inspector from "react-json-inspector";
import "./css/Details.css";

export interface IProps {
  classRepresentations: string[];
  end: Position;
  features: any;
  prediction: number;
  start: Position;
  rule: JSX.Element;
  y: number;
}

const Details = (props: IProps) => (
  <Grid fluid={true} className="Details">
    <Row>
      <Col sm={12}>
        <dl className="dl-horizontal">
          <dt>y</dt>
          <dd>
            {props.classRepresentations[props.y]} ({props.y})
          </dd>
          <dt>ŷ</dt>
          <dd>
            {props.prediction === -1
              ? "Not predicted"
              : `${props.classRepresentations[props.prediction]} (${
                  props.prediction
                })`}
          </dd>
          <dt>Rule</dt>
          <dd>{props.rule === null ? "None" : props.rule}</dd>
          <dt>Features</dt>
          <dd>
            <Inspector data={props.features} search={false} />
          </dd>
        </dl>
      </Col>
    </Row>
  </Grid>
);

export default Details;
