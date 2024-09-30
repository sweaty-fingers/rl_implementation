def is_best(best, compare, criterion: str):
    possible_criterion = ["min", "max"]
    if criterion not in possible_criterion:
        raise ValueError(f"criterion should be in {possible_criterion}\n Now get {criterion}")
    
    if criterion == "min":
        return compare < best
    else:
        return compare > best