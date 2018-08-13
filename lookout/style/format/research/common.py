from collections import defaultdict
import logging
from typing import Iterable, List

import bblfsh
from tqdm import tqdm

from lookout.style.format.features import WrappedNode


def prepare_nodes(uast: bblfsh.Node):
    nodes = {}
    root = WrappedNode.wrap(uast, None)
    stack = [(root, uast)]
    while stack:
        parent, parent_uast = stack.pop()
        stack.extend(zip((WrappedNode.wrap(child, parent) for child in parent_uast.children),
                         parent_uast.children))
        nodes[id(parent_uast)] = parent

    return nodes


def order_nodes(uast, excluded_internal_roles):
    """
    Select nodes with (tokens or specific node types) and with correct pos information
    -> order by `start_position.offset`
    """
    log = logging.getLogger("order_nodes")
    nodes = []
    for node in prepare_nodes(uast).values():
        if ((node.node.token or node.node.internal_type in excluded_internal_roles) and
                (node.start != node.end)):
            nodes.append(node)
    nodes = sorted(nodes, key=lambda n: n.start)

    # check overlapped nodes - it could happen because of nodes with exception_types
    def check_overlap(n1, n2):
        if n1.start <= n2.start < n1.end:
            if not n1.start <= n2.end and n1.end >= n2.end:
                log.debug("Required check: n1.start %d, n1.end %d, n2.start %d, n2.end %d" % (
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


def transform_content(content: str, uast: bblfsh.Node, filler,
                      excluded_internal_roles):
    """
    Visualize code without nodes with token and positions and fill theirs positions with filler.

    :param content: content.
    :param uast: UAST of content.
    :param filler: string that is used to fill the nodes.
    :param excluded_internal_roles: internal types that require special handling.

    :return: updated content.
    """
    nodes, _ = order_nodes(uast, excluded_internal_roles=excluded_internal_roles)

    # replace tokens with filler
    def insert_into_str(c, start, end):
        return c[:start] + "".join([filler] * (end - start)) + c[end:]

    for node in nodes:
        start = node.start
        end = node.end
        content = insert_into_str(content, start, end)
    return content


def _token_to_seq(token, to_check: Iterable[str]):
    if not token:
        return
    for check in to_check:
        pos = token.find(check)
        if pos != -1:
            left = token[:pos]
            center = token[pos:pos + len(check)]
            right = token[pos + len(check):]

            assert center == check, "%s != %s" % (center, check)
            report = (
                "Something wrong: token `%s`, left `%s`, center `%s`, right `%s`, check `%s`."
                % (token, left, center, right, check)
            )
            assert left + center + right == token, report

            yield from _token_to_seq(left, to_check)
            yield center
            yield from _token_to_seq(right, to_check)
            break


def token_to_seq(token, to_check: Iterable[str]):
    res = [n for n in _token_to_seq(token, to_check) if n is not None]
    # sanity check
    report = "token_to_seq: `%s` != `%s`" % ("".join(res), token)
    assert "".join(res) == token, report
    return res


def split_whitespaces_reserved(text, reserved_tokens: Iterable[str]):
    """
    Split text into whitespaces(including newlines/etc) and reserved keywords/operators.

    :param text: text with whitespaces and reserved keywords.
    :param reserved_tokens: list of reserved keywords and operators.
    :return: list of operators and whitespaces.
    """
    seq = text.split()

    # pure whitespaces
    if len(seq) == 0 and len(text) != 0:
        return list(text)

    # pure operators
    if len(seq) == 1 and len(text) == len(seq[0]):
        return token_to_seq(seq[0], reserved_tokens)

    # augment sequence with start and end position
    curr_pos = 0
    new_seq = []
    for el in seq:
        start = text.find(el, curr_pos)
        end = start + len(el)
        new_seq.append((start, end, el))
        curr_pos = end

    seq = new_seq
    if len(seq) == 0:
        if len(text) == 0:
            return []
        raise ValueError

    # mixed
    res = []
    # before first element
    if seq[0][0] != 0:
        res.append(text[:seq[0][0]])
    # between elements
    if len(seq) > 1:
        for el1, el2 in zip(seq[:-1], seq[1:]):
            res.extend(token_to_seq(el1[2], reserved_tokens))
            el1_end = el1[1]
            el2_start = el2[0]
            if el1_end != el2_start:
                res.append(text[el1_end:el2_start])
        res.extend(token_to_seq(el2[2], reserved_tokens))
    else:
        # add element in the end of sequence
        res.extend(token_to_seq(seq[0][2], reserved_tokens))

    # after last element
    last_end = seq[-1][1]
    if last_end != len(text):
        res.extend(text[last_end:])

    # sanity check
    report = "split_whitespaces_reserved: `%s` != `%s`, debug: %s" % (
        text, "".join(res), [token_to_seq(el[2], reserved_tokens) for el in seq]
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
                                        reserved_tokens: Iterable[str]):
    seq = []
    res = split_whitespaces_reserved(text=content[start:end], reserved_tokens=reserved_tokens)
    for t in res:
        end = start + len(t)
        n_new_lines = t.count("\n")
        end_line = start_line + n_new_lines
        if n_new_lines != 0:
            end_col = start_col + len(t) - (t.rfind("\n") + 1)
        else:
            end_col = start_col + len(t)

        n = WrappedNode(start=start, end=end, node=t, parent=common_anc,
                        start_line=start_line, start_col=start_col,
                        end_line=end_line, end_col=end_col)
        seq.append(n)
        start, start_col, start_line = end, end_col, end_line
    return seq


def extract_nodes(content, uast, reserved_tokens: Iterable[str],
                  excluded_internal_roles: Iterable[str]):
    """
    Extract list of Nodes ordered by position.
    :param content: content or text of source code.
    :param uast: UAST extracted from source code.
    :param reserved_tokens: list of reserved words ordered by length.
    :param excluded_internal_roles: list of exceptional internal types - special handling for them.
    :return: list of nodes.
    """
    uast_nodes, _ = order_nodes(uast, excluded_internal_roles=excluded_internal_roles)
    if len(uast_nodes) == 0:
        return

    # create new sequence of nodes from UAST and whitespace/operators/etc.
    seq = []
    if uast_nodes[0].start != 0:
        res = split_whitespaces_reserved_to_nodes(
            start=0, start_line=1, start_col=1, end=uast_nodes[0].start, content=content,
            common_anc=uast_nodes[0].parent, reserved_tokens=reserved_tokens
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
                common_anc=common_anc, reserved_tokens=reserved_tokens
            )
            seq.extend(res)

    seq.append(uast_nodes[-1])
    if uast_nodes[-1].end != len(content):
        res = split_whitespaces_reserved_to_nodes(
            start=uast_nodes[-1].end, start_line=uast_nodes[-1].start_line,
            start_col=uast_nodes[-1].start_col, end=len(content), content=content,
            common_anc=uast_nodes[-1].parent, reserved_tokens=reserved_tokens
        )
        seq.extend(res)

    # sanity check
    new_content = "".join(content[n.start:n.end] for n in seq)
    report = "extract_features: `%s` != `%s`" % (content, new_content)
    assert new_content == content, report

    return seq


def collect_unique_features(contents, uasts, reserved_tokens: Iterable[str],
                            excluded_internal_roles: Iterable[str], filenames: Iterable[str],
                            ignore_errors: bool=False):
    report = ("Number of contents (%d) & UASTs (%d) is not equal - something wrong."
              % (len(contents), len(uasts)))
    assert len(contents) == len(uasts), report
    internal_types = defaultdict(int)
    unique_features = defaultdict(int)
    for check in reserved_tokens:
        unique_features[check] += 1  # dummy counter for default reserved tokens
    for filename, content, uast in tqdm(zip(filenames, contents, uasts)):
        try:
            res = extract_nodes(content, uast, reserved_tokens,
                                excluded_internal_roles=excluded_internal_roles)
            if res is None:
                continue
            for el in res:
                if isinstance(el.node, str):
                    unique_features[el.node] += 1
                else:
                    unique_features[el.node.internal_type] += 1
            for node in prepare_nodes(uast).values():
                internal_types[node.node.internal_type] += 1
        except (AssertionError, ValueError):
            report = "Something wrong with file `%s`" % filename
            if ignore_errors:
                logging.warning(report)
                continue
            logging.error(report)
            raise
    logging.debug("Number of unique features: %s" % len(unique_features))
    return unique_features, internal_types


def extract_features(filenames: Iterable[str], contents: List[str],
                     uasts: List[bblfsh.Node], reserved_tokens: Iterable[str],
                     excluded_internal_roles: Iterable[str], seq_len: int=5, depth: int=5,
                     unique_features: Iterable[str]=None, use_features_after: bool=True,
                     use_parents: bool=True, ignore_errors: bool=False, use_siblings: bool=False):
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
    :param reserved_tokens: list of reserved tokens.
    :param excluded_internal_roles: list of exceptional internal types - special handling for them.
    :param seq_len: sequence length for features (before and after).
    :param depth: how many parents to use.
    :param unique_features: list of unique features. If None it will be collected from data.
    :param use_features_after: if context after label should be used.
    :param use_parents: if context about parent nodes should be used.
    :param ignore_errors: if ignore_errors than files with problems will be skipped.
    :param use_siblings: if context about siblings nodes should be used.
    :return: list of features, list of labels, list of metadata.
    """
    if unique_features is None:
        res = collect_unique_features(contents, uasts, reserved_tokens=reserved_tokens,
                                      ignore_errors=ignore_errors, filenames=filenames,
                                      excluded_internal_roles=excluded_internal_roles)
        unique_features, internal_types = res
        logging.debug("Number of unique features: {}".format(len(unique_features)))
        logging.debug("Number of unique internal types: {}".format(len(internal_types)))
    feature2id = dict((feat, i) for i, feat in enumerate(sorted(unique_features)))
    it2id = dict((internal_type, i) for i, internal_type in enumerate(sorted(internal_types)))

    def get_feature_id(feature):
        if isinstance(feature.node, str):
            return feature2id[feature.node]
        return feature2id.setdefault(feature.node.internal_type, len(feature2id))

    def _extract_features(element, use_pos=True, use_len=False):
        res = [get_feature_id(element)]
        if use_pos:
            res += [element.start, element.end]
        if use_len:
            res += [element.end - element.start]

        if use_parents:
            parents = []
            parent = element.parent
            for i in range(depth):
                parents.append(get_feature_id(parent))
                if parent.parent is not None:
                    parent = parent.parent
            res += parents

        if use_siblings:
            siblings = [0] * len(internal_types)
            parent = element.parent
            for sibling in parent.node.children:
                siblings[it2id[sibling.internal_type]] += 1
            res += siblings

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
        try:
            res = extract_nodes(content, uast, reserved_tokens=reserved_tokens,
                                excluded_internal_roles=excluded_internal_roles)
            if res is None:
                continue

            for i in range(len(res) - (2 * seq_len + 1)):
                min_pos = res[i].start
                feat = []

                # features before
                for j in range(seq_len):
                    feat.extend(_extract_features(res[i + j]))

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
                        feat.extend(_extract_features(res[i + j], use_pos=False, use_len=True))

                max_pos = res[i + j + 1].end
                metadata.append((file, min_pos, max_pos, raw_label.start, raw_label.end))

                features.append(feat)
                labels.append(label)
        except (AssertionError, ValueError):
            logging.error("Something wrong with file `{}`".format(file))
            if ignore_errors:
                continue
            raise

    return features, labels, metadata, feature2id
