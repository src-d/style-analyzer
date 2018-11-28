"""Javascript specific fixes for babelfish UASTs."""
import bblfsh


def fix_regexp_node(node: bblfsh.Node) -> bblfsh.Node:
    """
    Workaround https://github.com/bblfsh/javascript-driver/issues/37.

    Should be removed as soon as issue closed and new driver is used.
    """
    node.token = node.properties["pattern"]
    return node


NODE_FIXTURES = {
    "RegExpLiteral": fix_regexp_node,
}
