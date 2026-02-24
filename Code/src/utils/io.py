import os
import json

def get_paths():
    curr_dir = os.getcwd()
    base_path = os.path.abspath(os.path.join(curr_dir, os.pardir))
    code_path = os.path.join(base_path, "Code")
    data_path = os.path.join(base_path, "Dataset")
    output_path = os.path.join(base_path, "Outputs")
    print(base_path, code_path, data_path, output_path)
    return code_path, data_path, output_path

def read_json_file(input_path: str):
    print(f"Reading JSON file from: {input_path}")
    with open(input_path, "r") as f:
        content = f.read().strip()
        try:
            obj = json.loads(content)
            if isinstance(obj, list):
                return obj
            return obj
        except json.JSONDecodeError:
            pass

    with open(input_path, "r") as f:
        lines = f.readlines()
    parsed = [json.loads(l) for l in lines if l.strip()]
    if len(parsed) == 1 and isinstance(parsed[0], list):
        return parsed[0]
    return parsed


def save_json_file(data, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)
    print(f"Data saved to: {output_path}")


