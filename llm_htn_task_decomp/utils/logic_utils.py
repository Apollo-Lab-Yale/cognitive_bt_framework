def parse_instantiated_predicate(predicate_str):
    parts = predicate_str.split()
    predicate_name = parts[0]
    parameters = parts[1:]
    return predicate_name, parameters