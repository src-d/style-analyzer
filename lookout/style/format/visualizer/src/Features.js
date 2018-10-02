import React, { Component } from 'react';
import './css/Features.css';

class Features extends Component {

  render() {
    let toRender = null;
    if (this.props.features === null) {
      toRender = <span>No features to display.</span>
    } else {
      const features = this.props.features
      toRender = (
        <div className="row">
          <div className="col-sm">
            <dl>
              <dt>Start</dt>
              <dd>{`${features.start.offset}, ${features.start.line}, ${features.start.col}`}</dd>
              <dt>End</dt>
              <dd>{`${features.end.offset}, ${features.end.line}, ${features.end.col}`}</dd>
              <dt>Internal type</dt>
              <dd>{features.internal_type !== null ? features.internal_type : "None"}</dd>
              <dt>Class representation</dt>
              <dd>{features.y === null
                   ? features.value
                   : this.props.classRepresentations[features.y]}</dd>
              <dt>y</dt>
              <dd>{features.y === null ? "None" : features.y}</dd>
            </dl>
          </div>
        </div>
      );
    }
    return (
      <div className="Features">
        {toRender}
      </div>
    );
  }
}

export default Features;
