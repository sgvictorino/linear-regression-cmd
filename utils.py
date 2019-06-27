"""
Returns a list's first item that is not equal to None.
"""


def get_first_not_none_item_in_sequence(seq):
    for item in seq:
        if item is not None:
            return item


"""
Calls the given function with the given arguments, only if the provided value is not equal to None.
"""


def call_function_with_args_if_value_not_none(func, value, *args):
    if value is not None:
        return func(*args)
    return None
