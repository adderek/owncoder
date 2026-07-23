from tree import count_nodes


def test_single_node():
    assert count_nodes({"value": 1, "children": []}) == 1


def test_nested_tree():
    tree = {
        "value": 1,
        "children": [
            {"value": 2, "children": []},
            {"value": 3, "children": [
                {"value": 4, "children": []},
            ]},
        ],
    }
    assert count_nodes(tree) == 4
