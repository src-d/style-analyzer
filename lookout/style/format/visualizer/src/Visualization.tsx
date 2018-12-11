import zip from "lodash-es/zip";
import memoizeOne from "memoize-one";
import * as React from "react";
import { Col, Glyphicon, Grid, Panel, Row } from "react-bootstrap";
import ContextToken from "./ContextToken";
import "./css/Visualization.css";
import Details from "./Details";
import EmptyDetails from "./EmptyDetails";
import SampleToken from "./SampleToken";

export interface IState {
  absoluteConfidence: number;
  enabledRules: number[];
  highlighted?: number;
  neighbours: number[];
  relativeConfidence: number;
  support: number;
}

export interface IProps {
  data: Data;
  switchHandler: () => void;
}

class Visualization extends React.Component<IProps, IState> {
  private computeEnabledRulesSupport = memoizeOne(
    (support: number): boolean[] =>
      this.props.data.rules.map(rule => rule.support >= support)
  );

  private computeEnabledRulesAbsoluteConfidence = memoizeOne(
    (conf: number): boolean[] =>
      this.props.data.rules.map(rule => rule.conf >= conf)
  );

  private computeEnabledRulesRelativeConfidence = memoizeOne(
    (relativeConfidence: number): boolean[] => {
      const rules_by_confidence = this.props.data.rules_by_confidence;
      const enabledRules = rules_by_confidence.map(_ => true);
      for (
        let i = 0;
        i < this.props.data.rules.length - relativeConfidence;
        i++
      ) {
        enabledRules[rules_by_confidence[i]] = false;
      }
      return enabledRules;
    }
  );

  private computeEnabledRules = memoizeOne(
    (
      relativeConfidence: number,
      absoluteConfidence: number,
      support: number
    ): boolean[] => {
      const data = this.props.data;
      const enabledRulesSupport = this.computeEnabledRulesSupport(support);
      const enabledRulesAbsoluteConfidence = this.computeEnabledRulesAbsoluteConfidence(
        absoluteConfidence
      );
      const enabledRulesRelativeConfidence = this.computeEnabledRulesRelativeConfidence(
        relativeConfidence
      );
      const enabledRules = [];
      for (let i = 0; i < data.rules.length; i++) {
        enabledRules.push(
          enabledRulesSupport[i] &&
            enabledRulesAbsoluteConfidence[i] &&
            enabledRulesRelativeConfidence[i]
        );
      }
      return enabledRules;
    }
  );

  private computeNEnabledRules = memoizeOne(
    (enabledRules: boolean[]): number => enabledRules.filter(Boolean).length
  );

  private computeEnabled = memoizeOne(
    (enabledRules: boolean[]): boolean[] => {
      const result = [];
      const { break_uast, winners } = this.props.data;
      for (let i = 0; i < this.props.data.winners.length; i++) {
        result.push(!break_uast[i] && enabledRules[winners[i]]);
      }
      return result;
    }
  );

  private computeNEnabled = memoizeOne(
    (enabled: boolean[]): number => enabled.filter(Boolean).length
  );

  private computeCorrects = memoizeOne(
    (enabled: boolean[]): boolean[] => {
      const ys = this.props.data.ground_truths;
      const predictions = this.props.data.predictions;
      return enabled.map(
        (enabledi, index) => enabledi && ys[index] === predictions[index]
      );
    }
  );

  private computePrecision = memoizeOne(
    (enabled: boolean[], corrects: boolean[], nEnabled: number): number => {
      return (
        zip(enabled, corrects)
          .map(([enabledi, correct]) => enabledi && correct)
          .filter(Boolean).length / nEnabled
      );
    }
  );

  constructor(props: IProps) {
    super(props);
    this.state = {
      absoluteConfidence: 0,
      enabledRules: [],
      highlighted: undefined,
      neighbours: [],
      relativeConfidence: props.data.rules.length,
      support: 0
    };
  }

  public render() {
    const {
      absoluteConfidence,
      highlighted,
      neighbours,
      relativeConfidence,
      support
    } = this.state;
    const {
      class_printables,
      class_representations,
      features,
      ground_truths,
      labeled_indices,
      predictions,
      rule_uls,
      vnodes,
      winners
    } = this.props.data;
    const enabledRules = this.computeEnabledRules(
      relativeConfidence,
      absoluteConfidence,
      support
    );
    const nEnabledRules = this.computeNEnabledRules(enabledRules);
    const enabled = this.computeEnabled(enabledRules);
    const nEnabled = this.computeNEnabled(enabled);
    const corrects = this.computeCorrects(enabled);
    const precision = this.computePrecision(enabled, corrects, nEnabled);
    const highlightedLabeledIndex =
      highlighted === undefined ? undefined : labeled_indices.get(highlighted);
    const tokensByLines: JSX.Element[][] = [];
    let currentLine: JSX.Element[] = [];
    vnodes.forEach((vnode, index) => {
      const labeledIndex = labeled_indices.get(index);
      const value =
        labeledIndex === undefined
          ? vnode.value
          : class_printables[ground_truths[labeledIndex]];
      const lines = value.split(/\n|(?:\r\n)/);
      lines.forEach((line, i) => {
        currentLine.push(
          labeledIndex !== undefined ? (
            <SampleToken
              key={`${index}-${i}`}
              index={index}
              highlighted={highlighted === index}
              neighbour={neighbours.includes(index)}
              correct={corrects[labeledIndex]}
              value={line}
              enabled={enabled[labeledIndex]}
              highlightCallback={this.highlight}
            />
          ) : (
            <ContextToken
              key={`${index}-${i}`}
              neighbour={neighbours.includes(index)}
              value={line}
            />
          )
        );
        if (i < lines.length - 1) {
          tokensByLines.push(currentLine);
          currentLine = [];
        }
      });
    });
    if (currentLine.length > 0) {
      tokensByLines.push(currentLine);
    }
    const tokens = tokensByLines.map((lineTokens, index) => (
      <div key={index}>{lineTokens}</div>
    ));
    return (
      <Grid fluid={true} className="Visualization">
        <Row>
          <Col sm={12}>
            <Panel bsStyle="info">
              <Panel.Heading>
                <Glyphicon glyph="cog" /> Statistics and parameters
              </Panel.Heading>
              <Panel.Body>
                <Row>
                  <Col sm={4}>
                    Number of selected rules: {nEnabledRules} /{" "}
                    {enabledRules.length}
                  </Col>
                  <Col sm={4}>Precision: {precision.toFixed(2)}</Col>
                  <Col sm={4}>
                    Predicted Positive Condition Rate:{" "}
                    {(nEnabled / winners.length).toFixed(2)}
                  </Col>
                </Row>
                <Row>
                  <Col sm={12}>
                    <hr />
                  </Col>
                </Row>
                <Row>
                  <Col sm={4}>
                    Relative confidence threshold:{" "}
                    <input
                      id="confidence-relative"
                      type="number"
                      onChange={this.changeRelativeConfidence}
                      value={relativeConfidence}
                      min={0}
                      max={rule_uls.length}
                    />{" "}
                    / {rule_uls.length}
                  </Col>
                  <Col sm={4}>
                    Absolute confidence threshold{" "}
                    <input
                      id="confidence-absolute"
                      type="number"
                      onChange={this.changeAbsoluteConfidence}
                      value={absoluteConfidence}
                      min="0.0"
                      max="100.0"
                      step="0.1"
                    />{" "}
                    / 100%
                  </Col>
                  <Col sm={4}>
                    Support threshold{" "}
                    <input
                      id="support"
                      type="number"
                      onChange={this.changeSupport}
                      value={support}
                      min={0}
                    />
                  </Col>
                </Row>
              </Panel.Body>
            </Panel>
          </Col>
        </Row>
        <Row>
          <Col sm={6}>
            <Panel bsStyle="info">
              <Panel.Heading>
                <Glyphicon glyph="align-left" /> Code
              </Panel.Heading>
              <Panel.Body>
                <div className="Code">
                  <pre>
                    <code>{tokens}</code>
                  </pre>
                </div>
              </Panel.Body>
            </Panel>
          </Col>
          <Col sm={6}>
            <Panel bsStyle="info">
              <Panel.Heading>
                <Glyphicon glyph="th-list" /> Details
              </Panel.Heading>
              <Panel.Body>
                {highlightedLabeledIndex !== undefined ? (
                  <Details
                    start={vnodes[highlightedLabeledIndex].start}
                    end={vnodes[highlightedLabeledIndex].end}
                    y={ground_truths[highlightedLabeledIndex]}
                    rule={rule_uls[winners[highlightedLabeledIndex]]}
                    prediction={predictions[highlightedLabeledIndex]}
                    classRepresentations={class_representations}
                    features={features[highlightedLabeledIndex]}
                  />
                ) : (
                  <EmptyDetails />
                )}
              </Panel.Body>
            </Panel>
          </Col>
        </Row>
        <Row>
          <Col sm={12}>
            <button
              type="button"
              className="btn btn-primary btn-lg btn-block"
              onClick={this.props.switchHandler}
            >
              Try another input
            </button>
          </Col>
        </Row>
      </Grid>
    );
  }

  private findSiblings = (key: number): number[] => {
    const labeledIndex = this.props.data.labeled_indices.get(key);
    if (labeledIndex !== undefined) {
      return this.props.data.sibling_indices[labeledIndex];
    }
    return [];
  };

  private highlight = (key: number) => {
    if (key === this.state.highlighted) {
      this.setState({
        highlighted: undefined,
        neighbours: []
      });
    } else {
      this.setState({
        highlighted: key,
        neighbours: this.findSiblings(key)
      });
    }
  };

  private changeRelativeConfidence = (
    e: React.FormEvent<HTMLInputElement>
  ): void => {
    this.setState({
      relativeConfidence: parseInt(e.currentTarget.value, 10)
    });
    e.preventDefault();
  };

  private changeAbsoluteConfidence = (
    e: React.FormEvent<HTMLInputElement>
  ): void => {
    this.setState({
      absoluteConfidence: parseFloat(e.currentTarget.value)
    });
    e.preventDefault();
  };

  private changeSupport = (e: React.FormEvent<HTMLInputElement>): void => {
    this.setState({
      support: parseInt(e.currentTarget.value, 10)
    });
    e.preventDefault();
  };
}

export default Visualization;
