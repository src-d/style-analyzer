import os
from pathlib import Path
from typing import Any, Mapping, Optional

import pandas

from lookout.style.common import load_jinja2_template
from lookout.style.typos.analyzer import IDENTIFIER_INDEX_COLUMN, IdTyposAnalyzer, IdTyposModel
from lookout.style.typos.metrics import generate_report
from lookout.style.typos.utils import Candidate, Columns, flatten_df_by_column, TEMPLATE_DIR


TYPOS_DATASET = str(Path(__file__).parent / "data" / "commits_with_typo.csv.xz")


def evaluate_typos_on_identifiers(dataset: str = TYPOS_DATASET,
                                  config: Optional[Mapping[str, Any]] = None,
                                  mistakes_output: Optional[str] = None) -> str:
    """
    Run IdTyposAnalyzer on the identifiers from the evaluation dataset.

    :param dataset: Dataset of misspelled identifiers.
    :param config: Configuration for the IdTyposAnalyzer.
    :param mistakes_output: Path to the file for printing the wrong corrections.
    :return: Quality report.
    """
    identifiers = pandas.read_csv(dataset, header=0, usecols=[0, 1],
                                  names=["wrong", "correct"], keep_default_na=False)
    analyzer = IdTyposAnalyzer(IdTyposModel(), "", {} if config is None else config)
    suggestions = analyzer.check_identifiers(identifiers["wrong"].tolist())
    corrections = []
    for i, identifier in enumerate(identifiers["wrong"]):
        candidates = list(analyzer.generate_identifier_suggestions(suggestions[i], identifier))
        corrections.append(candidates if len(candidates) > 0 else [Candidate(identifier, 1.0)])

    for pos in range(analyzer.config["n_candidates"]):
        identifiers["sugg " + str(pos)] = [correction[pos][0] if pos < len(correction) else "" for
                                           correction in corrections]
    if mistakes_output is not None:
        identifiers[identifiers["sugg 0"] != identifiers["correct"]][
            ["wrong", "sugg 0", "correct"]].to_csv(mistakes_output)
    template = load_jinja2_template(os.path.join(TEMPLATE_DIR, "quality_on_identifiers.md.jinja2"))
    return template.render(identifiers=identifiers, suggestions=suggestions,
                           vocabulary_tokens=analyzer.corrector.generator.tokens,
                           n_candidates=analyzer.config["n_candidates"],
                           IDENTIFIER_INDEX_COLUMN=IDENTIFIER_INDEX_COLUMN,
                           Candidate=Candidate, Columns=Columns,
                           tokenize=lambda x: list(analyzer.parser.split(x)),
                           flatten_df_by_column=flatten_df_by_column,
                           generate_report=generate_report)
