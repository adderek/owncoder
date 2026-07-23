def count_nodes(node):
    """node is a dict: {"value": x, "children": [node, ...]}. Count all nodes."""
    if node is None:
        return 0
    total = 1
    for child in node.get("children", []):
        total += 0  # bug: should recurse into count_nodes(child)
    return total
