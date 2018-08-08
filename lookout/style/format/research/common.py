from collections import defaultdict
from copy import deepcopy
import logging

from tqdm import tqdm

from .js_reserved import JS_RESERVED


EXCEPTION_TYPES = ("RegExpLiteral", "TemplateLiteral")


class Node:
    def __init__(self, start, end, node, start_line=None, start_col=None, end_line=None,
                 end_col=None, parent=None):
        """
        :param start: start offset of node.
        :param end: end offset of node.
        :param node: UAST node or string with operator/reserved keywords/etc.
        :param start_line: start line of node (1-based index).
        :param start_col: start column of node (1-based index). Useful for calculation of
                          indentation.
        :param end_line: end line of node (line of newline character is counted as the same line as
                      start line) (1-based index).
        :param end_col: end column of node (1-based index).
        :param parent: parent node.
        """
        self.start = start
        if end != 0:
            self.end = end
        else:
            self.end = start + len(node.token) + 1

        self.node = node

        # start & end line/col are useful for nodes not from UAST
        self.start_line = start_line
        self.start_col = start_col
        self.end_line = end_line
        self.end_col = end_col

        self.parent = parent


def convert_node(node, parent):
    """
    Convert node from UAST to common format.

    :param node: UAST node
    :param parent: parent of this node
    :return: new node
    """
    return Node(start=node.start_position.offset, end=node.end_position.offset, node=node,
                start_line=node.start_position.line, start_col=node.start_position.col,
                end_line=node.end_position.line, end_col=node.end_position.col, parent=parent)


def prepare_nodes(uast):
    nodes = {}
    root = convert_node(uast, None)
    stack = [(root, uast)]
    while stack:
        parent, parent_uast = stack.pop()
        children_nodes = [convert_node(child, parent) for child in parent_uast.children]
        stack.extend(zip(children_nodes, parent_uast.children))
        nodes[id(parent_uast)] = parent

    return nodes


def ordered_nodes(uast, exception_types=EXCEPTION_TYPES):
    """
    Select nodes with (tokens or specific node types) and with correct pos information
    -> order by `start_position.offset`
    """
    log = logging.getLogger("ordered_nodes")
    nodes = []
    for node in prepare_nodes(uast).values():
        if ((node.node.token or node.node.internal_type in exception_types) and
                (node.start != node.end)):
            nodes.append(node)
    nodes = list(sorted(nodes, key=lambda n: n.start))

    # check overlapped nodes - it could happen because of nodes with exception_types
    def check_overlap(n1, n2):
        if (n1.start <= n2.start and n1.end > n2.start):
            if not (n1.start <= n2.end and n1.end >= n2.end):
                log.debug("Required check: n1.start {}, n1.end {}, n2.start {}, n2.end {}".format(
                    n1.start, n1.end, n2.start, n2.end))
            return True
        return False

    good_nodes = []
    curr_n = 0
    next_n = 1
    stop = False
    while not stop:
        if curr_n >= len(nodes):
            break
        good_nodes.append(curr_n)
        if next_n >= len(nodes):
            break
        while check_overlap(nodes[curr_n], nodes[next_n]):
            next_n += 1
            if next_n >= len(nodes):
                stop = True
                break
        curr_n = next_n
    return [nodes[n] for n in good_nodes], nodes


def transform_content(content: str, uast, filler: str="_", exception_types=EXCEPTION_TYPES):
    """
    Visualize code without nodes with token and positions and fill theirs positions with filler.

    :param content: content.
    :param uast: UAST of content.
    :param filler: string that is used to fill the nodes.
    :param exception_types: internal types that require special handling.

    :return: updated content.
    """
    nodes, _ = ordered_nodes(uast, exception_types=exception_types)
    content = deepcopy(content)

    # replace tokens with filler
    def insert_into_str(c, start, end):
        return c[:start] + "".join([filler] * (end - start)) + c[end:]

    for node in nodes:
        start = node.start
        end = node.end
        content = insert_into_str(content, start, end)
    return content


def _token_to_seq(token, to_check):
    if not token:
        return
    for check in JS_RESERVED:
        pos = token.find(check)
        if pos != -1:
            left = token[:pos]
            center = token[pos:pos + len(check)]
            right = token[pos + len(check):]

            assert center == check, "{} != {}".format(center, check)
            report = (
                "Something wrong: token `{}`, left `{}` , center `{}`, right `{}`, check `{}`."
                .format(token, left, center, right, check)
            )
            assert left + center + right == token, report

            yield from _token_to_seq(left, to_check)
            yield center
            yield from _token_to_seq(right, to_check)
            break


def token_to_seq(token, to_check):
    res = [n for n in _token_to_seq(token, to_check) if n is not None]
    # sanity check
    report = "token_to_seq: `{}` != `{}`".format("".join(res), token)
    assert "".join(res) == token, report
    return res


def split_whitespaces_reserved(text, reserved):
    """
    Split text into whitespaces(including newlines/etc) and reserved keywords/operators.

    :param text: text with whitespaces and reserved keywords.
    :param reserved: list of reserved keywords and operators.
    :return: list of operators and whitespaces.
    """
    seq = text.split()

    # pure whitespaces
    if len(seq) == 0 and len(text) != 0:
        return list(text)

    # pure operators
    if len(seq) == 1 and len(text) == len(seq[0]):
        return token_to_seq(seq[0], reserved)

    # augment sequence with start and end position
    curr_pos = 0
    new_seq = []
    for el in seq:
        start = text.find(el, curr_pos)
        end = start + len(el)
        new_seq.append((start, end, el))
        curr_pos = end

    seq = new_seq

    # mixed
    res = []
    # before first element
    if seq[0][0] != 0:
        res.append(text[:seq[0][0]])
    # between elements
    if len(seq) > 1:
        for el1, el2 in zip(seq[:-1], seq[1:]):
            res.extend(token_to_seq(el1[2], reserved))
            el1_end = el1[1]
            el2_start = el2[0]
            if el1_end != el2_start:
                res.append(text[el1_end:el2_start])
        res.extend(token_to_seq(el2[2], reserved))
    else:
        # add element in the end of sequence
        res.extend(token_to_seq(seq[0][2], reserved))

    # after last element
    last_end = seq[-1][1]
    if last_end != len(text):
        res.extend(text[last_end:])

    # sanity check
    report = "split_whitespaces_reserved: `{}` != `{}`, debug: {}".format(
        text, "".join(res), [token_to_seq(el[2], reserved) for el in seq]
    )
    assert "".join(res) == text, report
    return res


def find_common_ancestor(node1, node2):
    path_to_root = set()
    while node1.parent is not None:
        path_to_root.add(id(node1.parent))
        node1 = node1.parent

    while node2.parent is not None:
        if id(node2.parent) in path_to_root:
            # common ancestor is found
            return node2.parent
        node2 = node2.parent
    raise ValueError("Nodes don't have common ancestor!")


def split_whitespaces_reserved_to_nodes(start, start_line, start_col, end, common_anc, content,
                                        reserved):
    seq = []
    res = split_whitespaces_reserved(text=content[start:end], reserved=reserved)
    for t in res:
        end = start + len(t)
        n_new_lines = t.count("\n")
        end_line = start_line + n_new_lines
        if n_new_lines != 0:
            end_col = start_col + len(t) - (t.rfind("\n") + 1)
        else:
            end_col = start_col + len(t)

        n = Node(start=start, end=end, node=t, start_line=start_line, start_col=start_col,
                 end_line=end_line, end_col=end_col, parent=common_anc)
        seq.append(n)
        start, start_col, start_line = end, end_col, end_line
    return seq


def node_extraction(content, uast, reserved=JS_RESERVED):
    """
    Extract list of Nodes ordered by position.
    :param content: content or text of source code.
    :param uast: UAST extracted from source code.
    :param reserved: list of reserved words ordered by length.
    :return: list of nodes.
    """
    uast_nodes, _ = ordered_nodes(uast)
    if len(uast_nodes) == 0:
        return

    # create new sequence of nodes from UAST and whitespace/operators/etc.
    seq = []
    if uast_nodes[0].start != 0:
        res = split_whitespaces_reserved_to_nodes(
            start=0, start_line=1, start_col=1, end=uast_nodes[0].start, content=content,
            common_anc=uast_nodes[0].parent, reserved=reserved
        )
        seq.extend(res)

    for i in range(len(uast_nodes) - 1):
        seq.append(uast_nodes[i])
        if uast_nodes[i].end != uast_nodes[i + 1].start:
            start = uast_nodes[i].end
            start_line = uast_nodes[i].end_line
            start_col = uast_nodes[i].end_col
            end = uast_nodes[i + 1].start

            common_anc = find_common_ancestor(uast_nodes[i], uast_nodes[i + 1])
            res = split_whitespaces_reserved_to_nodes(
                start=start, start_line=start_line, start_col=start_col, end=end, content=content,
                common_anc=common_anc, reserved=reserved
            )
            seq.extend(res)

    seq.append(uast_nodes[-1])
    if uast_nodes[-1].end != len(content):
        res = split_whitespaces_reserved_to_nodes(
            start=uast_nodes[-1].end, start_line=uast_nodes[-1].start_line,
            start_col=uast_nodes[-1].start_col, end=len(content), content=content,
            common_anc=uast_nodes[-1].parent, reserved=reserved
        )
        seq.extend(res)

    # sanity check
    new_content = "".join(content[n.start:n.end] for n in seq)
    report = "feature_extraction: `{}` != `{}`".format(content, new_content)
    assert new_content == content, report

    return seq


def collect_unique_features(contents, uasts, reserved=JS_RESERVED):
    report = ("Number of contents ({}) & UASTs ({}) is not equal - something wrong."
              .format(len(contents), len(uasts)))
    assert len(contents) == len(uasts), report
    unique_features = defaultdict(int)
    for check in reserved:
        unique_features[check] += 1  # dummy counter for default reserved tokens
    for content, uast in tqdm(zip(contents, uasts)):
        res = node_extraction(content, uast, reserved)
        if res is None:
            continue
        for el in res:
            if isinstance(el.node, str):
                unique_features[el.node] += 1
            else:
                unique_features[el.node.internal_type] += 1
    logging.debug("Number of unique features: {}".format(len(unique_features)))
    return unique_features


def feature_extraction(filenames, contents, uasts, reserved=JS_RESERVED, seq_len=5, depth=5,
                       unique_features=None, use_features_after=True, use_parents=True):
    """
    Extract features:
    * before label
    * after label if `use_features_after`
    * information about parents if `use_parents`
    and extract label + metadata (filename, min & max position of features in code and position of
    label).

    :param filenames: list of filenames.
    :param contents: list of contents of files.
    :param uasts: list of extracted UASTs.
    :param reserved: list of reserved tokens.
    :param seq_len: sequence length for features (before and after).
    :param depth: how many parents to use.
    :param unique_features: list of unique features. If None it will be collected from data.
    :param use_features_after: if context after label should be used.
    :param use_parents: if context about parent nodes should be used.
    :return: list of features, list of labels, list of metadata.
    """
    if unique_features is None:
        unique_features = collect_unique_features(contents, uasts, reserved=reserved)
    feature2id = dict((feat, i) for i, feat in enumerate(sorted(unique_features)))

    def get_feature_id(feature):
        if isinstance(feature.node, str):
            return feature2id[feature.node]
        return feature2id.setdefault(feature.node.internal_type, len(feature2id))

    def extract_features(element, return_pos=True):
        parents = []
        if use_parents:
            parent = element.parent
            for i in range(depth):
                parents.append(get_feature_id(parent))
                if parent.parent is not None:
                    parent = parent.parent
        if return_pos:
            res = [element.start, element.end, get_feature_id(element)] + parents
            return res
        res = [get_feature_id(element)] + parents
        return res

    def count_beginning_spaces(line):
        if not line.startswith(" "):
            return 0
        for i, ch in enumerate(line):
            if ch != " ":
                return i + 1
        raise ValueError("It should not reach this point")

    label2id = {"nope": 0,
                "whitespace": 1,
                "newline": 2,
                "newline_incr": 3,
                "newline_decr": 4}

    features = []
    labels = []
    metadata = []

    for file, content, uast in tqdm(zip(filenames, contents, uasts)):
        res = node_extraction(content, uast, reserved)
        if res is None:
            continue

        for i in range(len(res) - (2 * seq_len + 1)):
            min_pos = res[i].start
            feat = []

            # features before
            for j in range(seq_len):
                feat.extend(extract_features(res[i + j]))

            # label
            label_ind = i + j + 1
            raw_label = res[label_ind]
            if (not isinstance(raw_label.node, str) or
                    ("\n" not in raw_label.node and " " not in raw_label.node)):
                label = label2id["nope"]
            elif " " in raw_label.node and "\n" not in raw_label.node:
                label = label2id["whitespace"]
            else:
                splitted_content = content.split("\n")

                start_line = splitted_content[raw_label.start_line - 1]
                end_line = splitted_content[raw_label.end_line - 1]
                start_cnt = count_beginning_spaces(start_line)
                end_cnt = count_beginning_spaces(end_line)
                if start_cnt == end_cnt:
                    label = label2id["newline"]
                elif start_cnt > end_cnt:
                    label = label2id["newline_decr"]
                else:
                    label = label2id["newline_incr"]

            if use_features_after:
                # features after
                for j in range(seq_len + 1, 2 * seq_len + 1):
                    assert label_ind != i + j, "Information leakage - label in features!"
                    feat.extend(extract_features(res[i + j], return_pos=False))

            max_pos = res[i + j + 1].end
            metadata.append((file, min_pos, max_pos, raw_label.start, raw_label.end))

            features.append(feat)
            labels.append(label)

    return features, labels, metadata
