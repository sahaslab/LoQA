import json
import string

# ----------------------------
# Normalization helpers
# ----------------------------

def normalize_string(s: str) -> str:
    """Lowercase, strip, remove punctuation, and fix whitespace"""
    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_punc(lower(s.strip())))


def list_normalization(lst):
    """
    Normalize each string and deduplicate
    Safe if lst is None or a single string.
    """
    if lst is None:
        return []
    if isinstance(lst, str):
        lst = [lst]

    normalized_lst = []
    for l in lst:
        # be robust if l isn't a string
        if l is None:
            continue
        normalized_lst.append(normalize_string(str(l)))

    normalized_lst = list(set(normalized_lst))
    return normalized_lst



# ----------------------------
# Null handling 
# ----------------------------

def is_nullish(x) -> bool:
    """
    Treat these as 'no prediction':
      - None
      - "null" (any case, surrounding whitespace ok)
      - "none"
      - empty string
    """
    if x is None:
        return True
    if isinstance(x, str) and x.strip().lower() in {"null", "none", ""}:
        return True
    return False


# ----------------------------
# Fallback extraction
# ----------------------------

def find_argument_value(role, string_to_check):
    """
    Extract argument value(s) for `role` from a substring of a JSON-ish object.
    Your heuristic kept, but:
      - default is []
      - trims split parts
      - filters nullish parts
    """
    role_index = string_to_check.find(role)
    if role_index == -1:
        return []

    if role_index == 0:
        return []

    finding_char = string_to_check[role_index - 1]
    start_search_index = role_index + len(role) + 2

    first_index = string_to_check.find(finding_char, start_search_index) + 1

    newline_index = string_to_check.find("\n", start_search_index)
    comma_index = string_to_check.find(",", start_search_index)
    if newline_index == -1:
        newline_index = float("inf")
    if comma_index == -1:
        comma_index = float("inf")
    newline_comma_index = min(newline_index, comma_index)

    if newline_comma_index <= first_index or first_index == 0:
        return []

    end_index = string_to_check.find(finding_char, first_index)
    value = string_to_check[first_index:end_index]

    # preserve split behavior, but trim + filter nullish
    parts = [p.strip() for p in value.split(";")]
    parts = [p for p in parts if not is_nullish(p)]
    return parts


def handle_json_decode_error(raw_pred, role):
    """
    Your fallback wrapper; returns [] when nothing extractable.
    """
    if is_nullish(raw_pred):
        return []

    start_index = raw_pred.find("{") + 1
    end_index = raw_pred.find("}", start_index)
    if start_index <= 0 or end_index <= start_index:
        return []

    string_of_interest = raw_pred[start_index:end_index]
    return find_argument_value(role, string_of_interest)


# ----------------------------
# Main processing
# ----------------------------

def get_process_predictions(raw_pred, role):
   
    if is_nullish(raw_pred):
        return []

    try:
        data_dict = json.loads(raw_pred)

        # IMPORTANT: select only the value for the role
        value = data_dict.get(role, None)
        if is_nullish(value):
            return []

        # split + clean
        parts = [p.strip() for p in str(value).split(";")]
        parts = [p for p in parts if not is_nullish(p)]
        return parts

    except Exception:
        # fallback to your heuristic extraction
        return handle_json_decode_error(raw_pred, role)

# print(get_process_predictions(None, "treatment"))          # should be []
# print(get_process_predictions('{"treatment":"null"}', "treatment"))  # should be []
# print(list_normalization(get_process_predictions('{"treatment":"null"}', "treatment")))  # should be []

# ----------------------------
# My old implementations
# ----------------------------

# import json
# import string


# # function for normalizing the strings
# def normalize_string(s):
#     def white_space_fix(text):
#         return " ".join(text.split())

#     def remove_punc(text):
#         exclude = set(string.punctuation)
#         return "".join(ch for ch in text if ch not in exclude)

#     def lower(text):
#         return text.lower()

#     return white_space_fix(remove_punc(lower(s.strip())))


# # this will be used for normalizing the actual and predicted labels
# def list_normalization(lst):
#     normalized_lst = []
#     for l in lst:
#         normalized_lst.append(normalize_string(l))
#     normalized_lst = list(set(normalized_lst))
#     return normalized_lst


# # extracting the argument value from the string when json decode error arises
# def find_argument_value(role, string_to_check):
#     value_list = ["null"]
#     role_index = string_to_check.find(role)

#     if role_index == -1:
#         return value_list

#     if role_index == 0:
#         return value_list

#     finding_char = string_to_check[role_index - 1]
#     start_search_index = role_index + len(role) + 2

#     first_index = string_to_check.find(finding_char, start_search_index) + 1

#     newline_index = string_to_check.find("\n", start_search_index)
#     comma_index = string_to_check.find(",", start_search_index)
#     if newline_index == -1:
#         newline_index = float("inf")
#     if comma_index == -1:
#         comma_index = float("inf")
#     newline_comma_index = min(newline_index, comma_index)

#     if newline_comma_index <= first_index or first_index == 0:
#         return ["null"]

#     end_index = string_to_check.find(finding_char, first_index)
#     value = string_to_check[first_index:end_index]
#     value_list = value.split(";")

#     return value_list


# # manually defining the function to extract values
# def handle_json_decode_error(raw_pred, role):
#     start_index = raw_pred.find("{") + 1
#     end_index = raw_pred.find("}", start_index)
#     string_of_interest = raw_pred[start_index:end_index]
#     value_list = find_argument_value(role, string_of_interest)
#     return value_list


# # processing the predictions for each role
# def get_process_predictions(raw_pred, role):
#     flag = 0
#     value_list = ["null"]

#     try:
#         data_dict = json.loads(raw_pred)
#         for key, value in data_dict.items():
#             if value is None:
#                 value = "null"
#             value = str(value)
#             value_list = set(value.split(";"))
#     except (json.JSONDecodeError, KeyError, Exception):
#         flag = 1

#     if flag == 1:
#         value_list = handle_json_decode_error(raw_pred, role)

#     return list(value_list)