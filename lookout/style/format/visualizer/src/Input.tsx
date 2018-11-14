import * as React from "react";
import { Col, Grid, Row } from "react-bootstrap";
import "./css/Input.css";
import defaultCode from "./defaultCode";

export interface IProps {
  switchHandler: (code: string) => void;
}

const Input = (props: IProps) => {
  const code = React.useRef<HTMLTextAreaElement>(null);
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
            placeholder="Please enter the code here"
            ref={code}
            defaultValue={defaultCode}
          />
        </Col>
      </Row>
      <Row>
        <Col sm={12}>
          <button
            type="button"
            className="btn btn-primary btn-lg btn-block"
            onClick={() =>
              props.switchHandler(
                code.current !== null ? code.current.value : ""
              )
            }
          >
            Visualize
          </button>
        </Col>
      </Row>
    </Grid>
  );
};

export default Input;
