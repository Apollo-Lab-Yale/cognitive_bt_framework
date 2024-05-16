def parse_instantiated_predicate(predicate_str):
    parts = predicate_str.split()
    predicate_name = parts[0]
    target = parts[1] if len(parts) > 1 else None
    return predicate_name, target