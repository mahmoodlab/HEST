import datetime

def get_current_time():
    now = datetime.datetime.now()
    year = now.year % 100  # convert to 2-digit year
    month = now.month
    day = now.day
    hour = now.hour
    minute = now.minute
    second = now.second
    return f"{year:02d}-{month:02d}-{day:02d}-{hour:02d}-{minute:02d}-{second:02d}"


def merge_dict(main_dict, new_dict, value_fn = None):
    """
    Merge new_dict into main_dict. If a key exists in both dicts, the values are appended. 
    Else, the key-value pair is added.
    Expects value to be an array or list - if not, it is converted to a list.
    If value_fn is not None, it is applied to each item in each value in new_dict before merging.
    Args:
        main_dict: main dict
        new_dict: new dict
        value_fn: function to apply to each item in each value in new_dict before merging
    """
    if value_fn is None:
        value_fn = lambda x: x
    for key, value in new_dict.items():
        if not isinstance(value, list):
            value = [value]
        value = [value_fn(v) for v in value]
        if key in main_dict:
            main_dict[key] = main_dict[key] + value
        else:
            main_dict[key] = value
    return main_dict