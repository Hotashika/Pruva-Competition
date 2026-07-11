import json


def main(file_adress):
    # Read JSON file
    with open(file_adress, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Convert to list
    if isinstance(data, list):
        result = data
    elif isinstance(data, dict):
        result = list(data.items())
    else:
        result = [data]

    return result
