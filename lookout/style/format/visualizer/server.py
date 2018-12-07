"""Visualize features of the format analyzer."""
from collections.abc import Mapping as Mapping_abc
import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Union

from bblfsh import BblfshClient
from flask import abort, Flask, jsonify, request, Response
from flask_cors import CORS
from lookout.core.api.service_data_pb2 import File
import numpy
from scipy.sparse import csr_matrix

from lookout.style.format.descriptions import describe_rules, describe_sample
from lookout.style.format.feature_extractor import FeatureExtractor
from lookout.style.format.model import FormatModel
from lookout.style.format.postprocess import filter_uast_breaking_preds
from lookout.style.format.virtual_node import VirtualNode

logging.basicConfig(level="DEBUG")
app = Flask(__name__)
CORS(app)


def _convert_to_jsonable(mapping: MutableMapping[Any, Any]) -> MutableMapping[Any, Any]:
    for key, value in mapping.items():
        if isinstance(value, numpy.ndarray):
            mapping[key] = value.tolist()
        elif isinstance(value, Mapping_abc):
            mapping[key] = _convert_to_jsonable(value)
    return mapping


def _vnode_to_dict(vnode: VirtualNode) -> Mapping[str, Any]:
    return {
        "start": {"offset": int(vnode.start.offset),
                  "col": int(vnode.start.col),
                  "line": int(vnode.start.line)},
        "end": {"offset": int(vnode.end.offset),
                "col": int(vnode.end.col),
                "line": int(vnode.end.line)},
        "value": vnode.value,
        "path": vnode.path,
        "roles": [role for role in vnode.node.roles] if vnode.node else [],
        "y": vnode.y,
        "internal_type": vnode.node.internal_type if vnode.node else None,
    }


DictOrStr = Dict[str, Union[Dict[str, "DictOrStr"], str]]


def _input_matrix_to_descriptions(X_csr: csr_matrix, feature_extractor: FeatureExtractor
                                  ) -> List[DictOrStr]:
    X = X_csr.toarray()
    descriptions = []
    for x in X:
        offset = 0
        groups = {}
        for group, group_value in feature_extractor._features.items():
            groups[group.name] = []
            for node_index, node_value in enumerate(group_value):
                groups[group.name].append({})
                for feature_id, feature in node_value.items():
                    feature_values = x[offset:offset + len(feature.selected_names)]
                    offset += len(feature.selected_names)
                    description = describe_sample(feature, feature_values)
                    groups[group.name][node_index][feature_id.name] = description
        descriptions.append(groups)
    return descriptions


@app.route("/", methods=["POST"])
def return_features() -> Response:
    """Featurize the given code."""
    body = request.get_json()
    code = body["code"]
    babelfish_address = body["babelfish_address"]
    language = body["language"]
    client = BblfshClient(babelfish_address)
    res = client.parse(filename="", contents=code.encode(), language=language)
    if res.status != 0:
        abort(500)
    model = FormatModel().load(str(Path(__file__).parent / "models" / "model.asdf"))
    rules = model[language]
    file = File(content=code.encode(), uast=res.uast, language="javascript")
    config = rules.origin_config["feature_extractor"]
    config["return_sibling_indices"] = True
    fe = FeatureExtractor(language=language, **config)
    res = fe.extract_features([file])
    if res is None:
        abort(500)
    X, y, (vnodes_y, vnodes, vnode_parents, node_parents, sibling_indices) = res
    y_pred, rule_winners, rules = rules.predict(X=X, vnodes_y=vnodes_y, vnodes=vnodes,
                                                feature_extractor=fe)
    _, _, _, _, safe_preds = filter_uast_breaking_preds(
        y=y, y_pred=y_pred, vnodes_y=vnodes_y, vnodes=vnodes, files={file.path: file},
        feature_extractor=fe, stub=client._stub, vnode_parents=vnode_parents,
        node_parents=node_parents, rule_winners=rule_winners, log=app.logger)
    wrong_preds = list(set(range(X.shape[0])) - set(safe_preds))
    rule_winners[wrong_preds] = -1
    app.logger.info("returning features of shape %d, %d" % X.shape)
    app.logger.info("length of rules: %d", len(rules))
    return jsonify({"code": code,
                    "features": _input_matrix_to_descriptions(X, fe),
                    "ground_truths": y.tolist(),
                    "predictions": y_pred.tolist(),
                    "sibling_indices": sibling_indices,
                    "rules": describe_rules(rules.rules, fe),
                    "confidences": [float(rule.stats.conf * 100) for rule in rules.rules],
                    "supports": [int(rule.stats.support) for rule in rules.rules],
                    "winners": rule_winners.tolist(),
                    "feature_names": fe.feature_names,
                    "class_representations": fe.composite_class_representations,
                    "class_printables": fe.composite_class_printables,
                    "vnodes": list(map(_vnode_to_dict, vnodes)),
                    "config": _convert_to_jsonable(rules.origin_config)})
