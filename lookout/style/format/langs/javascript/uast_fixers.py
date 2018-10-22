"""Javascript specific fixes for babelfish UASTs."""
from bblfsh import Node


def fix_regexp_node(node: Node) -> Node:
    """
    Workaround https://github.com/bblfsh/javascript-driver/issues/37.

    Should be removed as soon as issue closed and new driver is used.
    """
    node.token = node.properties["pattern"]
    return node


NODE_FIXTURES = {
    "RegExpLiteral": fix_regexp_node,
}
