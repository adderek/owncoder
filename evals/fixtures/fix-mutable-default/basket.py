def add_item(name, items=[]):  # bug: mutable default arg shared across calls
    items.append(name)
    return items
