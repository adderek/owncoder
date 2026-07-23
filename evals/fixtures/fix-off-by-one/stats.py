def running_totals(values):
    """Return a list where result[i] = sum(values[0..i])."""
    out = []
    total = 0
    for i in range(len(values) - 1):  # bug: drops the last element
        total += values[i]
        out.append(total)
    return out
