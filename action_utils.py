# action_utils.py
import re
from json import loads, JSONDecodeError
from typing import Dict, Any

def parse_action_response(response_text: str) -> Dict[str, Any]:
    """Check if the AI’s response is in the right format."""
    try:
        # Step 1: Try to find a JSON object using regex
        json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if not json_match:
            print("Error: No JSON object found in the response.")
            return {}

        json_str = json_match.group(0)

        # Step 2: Sanitize triple quotes around SQL (replace """ with ")
        json_str = re.sub(r'""+"', '"', json_str)  # Handles """ and "" as well

        # Step 3: Replace any newlines within strings with \n safely
        def escape_multiline_strings(match):
            content = match.group(0)
            return content.replace('\n', '\\n')

        json_str = re.sub(r'".*?"', escape_multiline_strings, json_str, flags=re.DOTALL)

        # Step 4: Try parsing
        try:
            json_function = loads(json_str)
        except JSONDecodeError:
            # Handle common backslash-quote escape error (e.g., \' becoming '')
            fixed_json_str = json_str.replace("\\'", "''")
            try:
                json_function = loads(fixed_json_str)
            except JSONDecodeError:
                print("Error: Response is not valid JSON even after attempting a fix.")
                return {}

        # Step 5: Check required structure
        if not isinstance(json_function, dict) or "function_name" not in json_function or "function_parms" not in json_function:
            print("Error: Response is not in the expected format.")
            return {}

        return json_function

    except Exception as e:
        print(f"Error: {str(e)}")
        return {}
def execute_action(action: Dict[str, Any], available_actions: Dict[str, Any]) -> Dict[str, Any]:
    """Do what the AI says (execute the action)."""
    function_name = action.get("function_name")
    function_parms = action.get("function_parms")

    if function_name not in available_actions:
        return {"error": f"Unknown function name: {function_name}"}

    try:
        action_function = available_actions[function_name]
        return action_function(**function_parms)
    except Exception as e:
        return {"error": f"Error executing {function_name}: {str(e)}"}