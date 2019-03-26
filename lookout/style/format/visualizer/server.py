"""Visualize features of the format analyzer."""
from collections.abc import Mapping as Mapping_abc
from functools import partial
import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Union

from bblfsh import BblfshClient
from flask import abort, Flask, jsonify, request, Response
from flask_cors import CORS
from lookout.core.analyzer import UnicodeFile
import numpy
from scipy.sparse import csr_matrix
from sklearn.exceptions import NotFittedError

from lookout.style.format.descriptions import describe_rule_attrs, describe_sample
from lookout.style.format.feature_extractor import FeatureExtractor
from lookout.style.format.model import FormatModel
from lookout.style.format.rules import Rules
from lookout.style.format.uast_stability_checker import UASTStabilityChecker
from lookout.style.format.virtual_node import VirtualNode

logging.basicConfig(level="DEBUG")
app = Flask(__name__)
CORS(app)


def _mapping_to_jsonable(mapping: Mapping[Any, Any]) -> Mapping[Any, Any]:
    jsonable = {}
    for key, value in mapping.items():
        if isinstance(value, numpy.ndarray):
            jsonable[key] = value.tolist()
        elif isinstance(value, Mapping_abc):
            jsonable[key] = _mapping_to_jsonable(value)
    return jsonable


def _rules_to_jsonable(rules: Rules, feature_extractor: FeatureExtractor,
                       ) -> Sequence[Mapping[str, Any]]:
    return [dict(attrs=describe_rule_attrs(rule, feature_extractor),
                 cls=rule.stats.cls,
                 conf=rule.stats.conf * 100,
                 support=rule.stats.support,
                 artificial=rule.artificial)
            for rule in rules.rules]


def _vnode_to_jsonable(vnode: VirtualNode, labeled_indices: Mapping[int, int],
                       ) -> Mapping[str, Any]:
    jsonable = {
        "start": {"offset": int(vnode.start.offset),
                  "col": int(vnode.start.col),
                  "line": int(vnode.start.line)},
        "end": {"offset": int(vnode.end.offset),
                "col": int(vnode.end.col),
                "line": int(vnode.end.line)},
        "value": vnode.value,
        "path": vnode.path,
    }
    if vnode.y is not None:
        jsonable["y"] = vnode.y
    if id(vnode) in labeled_indices:
        jsonable["labeled_index"] = labeled_indices[id(vnode)]
    return jsonable


DictOrStr = Dict[str, Union[Dict[str, "DictOrStr"], str]]


def _input_matrix_to_descriptions(X_csr: csr_matrix, feature_extractor: FeatureExtractor,
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
    if language not in model:
        raise NotFittedError()
    rules = model[language]
    file = UnicodeFile(content=code, uast=res.uast, language="javascript", path="path")
    config = rules.origin_config["feature_extractor"]
    config["return_sibling_indices"] = True
    fe = FeatureExtractor(language=language, **config)
    res = fe.extract_features([file])
    if res is None:
        abort(500)
    X, y, (vnodes_y, vnodes, vnode_parents, node_parents, sibling_indices) = res
    y_pred, rule_winners, rules, grouped_quote_predictions = rules.predict(
        X=X, vnodes_y=vnodes_y, vnodes=vnodes, feature_extractor=fe)
    refuse_to_predict = y_pred < 0
    checker = UASTStabilityChecker(fe)
    _, _, _, _, safe_preds = checker.check(
        y=y, y_pred=y_pred, vnodes_y=vnodes_y, vnodes=vnodes, files=[file], stub=client._stub,
        vnode_parents=vnode_parents, node_parents=node_parents, rule_winners=rule_winners,
        grouped_quote_predictions=grouped_quote_predictions)
    break_uast = [False] * X.shape[0]
    for wrong_pred in set(range(X.shape[0])).difference(safe_preds):
        break_uast[wrong_pred] = True
    labeled_indices = {id(vnode): i for i, vnode in enumerate(vnodes_y)}
    app.logger.info("returning features of shape %d, %d" % X.shape)
    app.logger.info("length of rules: %d", len(rules))
    return jsonify({
        "code": code,
        "features": _input_matrix_to_descriptions(X, fe),
        "ground_truths": y.tolist(),
        "predictions": y_pred.tolist(),
        "refuse_to_predict": refuse_to_predict.tolist(),
        "sibling_indices": sibling_indices,
        "rules": _rules_to_jsonable(rules, fe),
        "winners": rule_winners.tolist(),
        "break_uast": break_uast,
        "feature_names": fe.feature_names,
        "class_representations": fe.composite_class_representations,
        "class_printables": fe.composite_class_printables,
        "vnodes": list(map(partial(_vnode_to_jsonable, labeled_indices=labeled_indices), vnodes)),
        "config": _mapping_to_jsonable(rules.origin_config)})
