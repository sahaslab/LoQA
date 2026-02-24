import json
import string

# function for normalizing the strings
def normalize_string(s):
      def white_space_fix(text):
          return ' '.join(text.split())
      def remove_punc(text):
          exclude = set(string.punctuation)
          return ''.join(ch for ch in text if ch not in exclude)
      def lower(text):
          return text.lower()
      return white_space_fix(remove_punc(lower(s.strip())))

# this will be used for normalizing the actual and predicted labels
def list_normalization(lst):
    normalized_lst = []
    for l in lst:
        normalized_lst.append(normalize_string(l))  # keep current normalization behavior
        # normalized_lst.append(l)  # keep as-is for your toggle
    normalized_lst = list(set(normalized_lst))  # preserves your dedup behavior (unordered)
    return normalized_lst

# extracting the argument value from the string when json decode error arises
def find_argument_value(role, string_to_check):
    value_list = ['null']  # default
    role_index = string_to_check.find(role)
    if role_index == -1:
        return value_list

    # Determine the encapsulating quote character just before role
    if role_index == 0:
        return value_list
    finding_char = string_to_check[role_index - 1]

    # Move start past 'role' and the following ': ' pattern heuristic (+2)
    start_search_index = role_index + len(role) + 2

    # First quote after role
    first_index = string_to_check.find(finding_char, start_search_index) + 1

    # Stop at nearest newline or comma after role (as in original heuristic)
    newline_index = string_to_check.find('\n', start_search_index)
    comma_index = string_to_check.find(',', start_search_index)
    if newline_index == -1:
        newline_index = float('inf')
    if comma_index == -1:
        comma_index = float('inf')
    newline_comma_index = min(newline_index, comma_index)

    # Handle null-like early termination or malformed indexing
    if newline_comma_index <= first_index or first_index == 0:
        return ['null']

    # Extract up to the closing quote
    end_index = string_to_check.find(finding_char, first_index)
    value = string_to_check[first_index:end_index]
    value_list = value.split(";")  # preserve original split behavior (no trimming)

    return value_list

# manually defining the function to extract values
def handle_json_decode_error(raw_pred, role):
    start_index = raw_pred.find('{') + 1
    end_index = raw_pred.find('}', start_index)

    string_of_interest = raw_pred[start_index:end_index]
    value_list = find_argument_value(role, string_of_interest)
    return value_list

# processing the predictions for each role
def get_process_predictions(raw_pred, role):
    flag = 0
    value_list = ['null']  # ensure defined even if json loads succeeds with empty dict
    try:
        data_dict = json.loads(raw_pred)
        for key, value in data_dict.items():
            if value == None:
                value = 'null'
            value = str(value)
            value_list = set(value.split(";")) # last item's value_list wins (same as original loop)
    except json.JSONDecodeError as e:
        flag = 1
    except KeyError as e:
        flag = 1
    except Exception as e:
        flag = 1

    # if any error occurred, fallback to manual extraction
    if flag == 1:
        value_list = handle_json_decode_error(raw_pred, role)

    return list(value_list)