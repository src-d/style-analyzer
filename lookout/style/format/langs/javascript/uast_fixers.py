"""Javascript specific fixes for babelfish UASTs."""
import bblfsh


def fix_string_literal_type_anotation(node: bblfsh.Node) -> bblfsh.Node:
    """
    Workaround https://github.com/bblfsh/javascript-driver/issues/66.

    Should be removed as soon as issue closed and new driver is used.
    """
    if node.token == "":
        node.token = node.properties["value"]
    return node


def fix_operator_node(node: bblfsh.Node) -> bblfsh.Node:
    """
    Workaround https://github.com/bblfsh/javascript-driver/issues/65.

    Should be removed as soon as issue closed and new driver is used.
    """
    if (node.start_position.offset + node.start_position.col + node.start_position.line +
            node.end_position.offset + node.end_position.col + node.end_position.line == 0):
        node.token = ""
    return node


NODE_FIXTURES = {
    "Operator": fix_operator_node,
    "StringLiteralTypeAnnotation": fix_string_literal_type_anotation,
}
