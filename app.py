import re
import subprocess
import os
import requests
import openai
import google.generativeai as genai
#import anthropic  # Import the official Anthropic client
from flask import Flask, request, jsonify
from flask_cors import CORS
import threading
import traceback
import html # For escaping result content
from playwright.sync_api import sync_playwright

# --- Flask App Setup ---
app = Flask(__name__)

CORS(app) # Enable CORS

# --- Global State (Simple Approach) ---
# No persistent history needed here if frontend sends full history each time
history_lock = threading.Lock() # Still good practice if Flask uses threads

# API Key Storage (Updated by requests, defaults to env vars)
current_api_keys = {
    "openai": os.getenv("OPENAI_API_KEY"),
    "gemini": os.getenv("GEMINI_API_KEY"),
    "claude": os.getenv("ANTHROPIC_API_KEY") # Anthropic client uses ANTHROPIC_API_KEY
}
# Hardcoded Gemini key only as a *last* resort if env var isn't set
GEMINI_API_KEY_HARDCODED = "AIzaSyACSOCAbT5iDb8gFtmnmvKBTUsbDMKumJ0" # Replace if needed, but env var is better

# Security: Restrict file operations
ALLOWED_DIR = os.getcwd()
print(f"--- File operations restricted to: {ALLOWED_DIR} ---")

# --- Core Prompting and Parsing ---

# ** MODIFIED TAG_INFO **
TAG_INFO = """
You are a multi puprpose agent running in a Windows environment. You have access to the following tools. Use them by outputting the corresponding XML-like tags directly in your response.

Available Tools:

- <TERMINAL>command</TERMINAL>: Executes the shell `command` and returns its standard output or error. Use for general commands. If you are asked to run a file, and the exact path is not given, try to find it using terminal commands to locate the file first.YOU KNOW HOW TO RUN FILES (just do it through the terminal like python name.py, etc for other stuff)
- <FILEWRITE path=\"path/to/file\">content_to_write</FILEWRITE>: Writes the exact `content_to_write` into the specified file `path`.
    - **IMPORTANT**: Place the complete, raw file content *directly* between the tags. Do NOT use markdown code blocks (```) inside the tag content.
    - The system will confirm success or report errors. Do not repeat the file content in your response after writing.
- <FILEREAD path=\"path/to/file\"></FILEREAD>: Reads the entire content of the file at `path` and returns it. Use this to check file contents. (If the file path is not explicitly given try to find it using terminal by listing all files and picking what is required)
- <WEBREQUEST url=\"[https://example.com](https://example.com)\"></WEBREQUEST>: Fetches the fully rendered HTML of a web page (including JavaScript content) using a headless browser. Falls back to a simple HTTP GET if browser automation is unavailable.

- <CONFIRM>Question for user?</CONFIRM>: Asks the user the 'Question for user?' and waits for an 'Approved' or 'Rejected' response. Use this *before* performing potentially destructive or permission-requiring actions (like installing packages, deleting files, running risky commands).

How Tool Interaction Works:

1.  You include one or more tool tags in your response (e.g., `<FILEWRITE path="code.py">print("Hello")</FILEWRITE>`).
2.  The system executes the requested action(s).
3.  The system will then send the results back to you in the *next* user message, formatted like this: `<[TOOL_NAME]_RESULT>result_content</[TOOL_NAME]_RESULT>`.
    - Example Success: `<FILEWRITE_RESULT>File 'code.py' written successfully</FILEWRITE_RESULT>`
    - Example Content: `<FILEREAD_RESULT>print("Hello World!")</FILEREAD_RESULT>`
    - Example Confirmation: `<CONFIRM_RESULT>Approved</CONFIRM_RESULT>`
    - Example Error: `<TERMINAL_RESULT>Error: command not found</TERMINAL_RESULT>`
4.  **Crucially:** You should **READ** the content of these `_RESULT` tags to understand what happened. **DO NOT** generate `<[TOOL_NAME]_RESULT>` tags or generic `<TAG_RESULT>` tags yourself. Integrate the information from the results naturally into your conversational reply to the user. For example, after a successful FILEREAD, you might say "Okay, here is the content of the file:" followed by the content you received in the `<FILEREAD_RESULT>` tag.

General Instructions:
- Ensure file paths are relative (e.g., `my_folder/script.py`). Operations are restricted to the current working directory.
- When writing code using FILEWRITE, ensure it's functional and complete *inside* the tag.
- Ask for confirmation (<CONFIRM>) before running commands that modify the system (install, delete, etc.) or execute potentially complex scripts.
- Stick *only* to the tags provided above. Do not invent new tags.
- ALWAYS REMEMBER U HAVE ACCESS TO THE TERMINAL SO U CAN RUN FILES AND STUFF. IF THE USER ASKS U TO RUN FILES DO NOT JUST ASK THEM TO DO IT THEMSELVES.  
"""

# Tag parsing (remains the same)
TAG_PATTERN = re.compile(r'<(\w+)([^>]*)>(.*?)</\1>', re.DOTALL)
ATTR_PATTERN = re.compile(r'(\w+)=(?:"([^"]*)"|([^\s>]*))')

def parse_tags(text):
    if not isinstance(text, str): return []
    matches = TAG_PATTERN.findall(text)
    tags = []
    for match in matches:
        tag_name, attr_str, content = match
        attributes = {}
        for attr_match in ATTR_PATTERN.findall(attr_str):
            attr_name, quoted_value, unquoted_value = attr_match
            attributes[attr_name] = quoted_value if quoted_value else unquoted_value
        # Ensure tag name is uppercase for consistent matching
        tags.append({"tag": tag_name.upper(), "attributes": attributes, "content": content}) # No strip here, keep whitespace for code
    return tags

# Function to clean code of markdown formatting (remains the same)
def clean_code_for_file(content):
    """Clean markdown code blocks and other formatting from content before writing to file."""
    if content is None:
        return ""
    
    # First, check if the entire content is wrapped in a code block
    pattern = r'^\s*```[\w.-]*\s*\n(.*?)\n\s*```\s*$'
    match = re.match(pattern, content, re.DOTALL)
    if match:
        return match.group(1).strip()  # Return just the code inside, trimmed
    
    # Otherwise clean any markdown code blocks anywhere in the content
    # Remove code block markers with language identifier
    content = re.sub(r'```[\w.-]*\s*\n', '', content)
    # Remove remaining code block markers
    content = re.sub(r'```', '', content)
    # Clean up any extra newlines that might have been left
    content = re.sub(r'\n{3,}', '\n\n', content)
    
    return content.strip()

# --- Tag Execution Functions (Added robust path checking) ---
def _resolve_safe_path(path):
    """Resolves a relative path safely within ALLOWED_DIR."""
    if not path or '..' in path or path.startswith('/'):
        # Basic checks against path traversal or absolute paths
        return None, f"Error: Invalid or potentially unsafe path '{path}'"

    try:
        abs_path = os.path.abspath(os.path.join(ALLOWED_DIR, path))
        if not abs_path.startswith(os.path.abspath(ALLOWED_DIR)):
             return None, f"Error: Path '{path}' resolves outside allowed directory."
        # Ensure parent directories exist for write/execute operations if needed later
        # os.makedirs(os.path.dirname(abs_path), exist_ok=True) # Do this in write/execute specifically
        return abs_path, None
    except Exception as e:
        return None, f"Error resolving path '{path}': {str(e)}"

def execute_terminal(attributes, content):
    print(f"Executing TERMINAL: {content}")
    try:
        # Use timeout, run in allowed dir
        content = clean_code_for_file(content)
        result = subprocess.run(content, shell=True, capture_output=True, text=True, timeout=60, cwd=ALLOWED_DIR, check=False) # check=False to capture stderr on failure
        output = result.stdout if result.returncode == 0 else f"Error (Return Code {result.returncode}): {result.stderr}"
        # Limit output size slightly? Maybe not for terminal.
        return output.strip()
    except subprocess.TimeoutExpired:
        return "Error: Command timed out after 60 seconds."
    except Exception as e:
        print(f"ERROR in execute_terminal: {traceback.format_exc()}")
        return f"Exception: {str(e)}"

def write_file(attributes, content):
    path = attributes.get("path")
    print(f"Executing FILEWRITE: {path}")
    if not path: return "Error: Missing path attribute"

    abs_path, error = _resolve_safe_path(path)
    if error: return error

    # ** Crucial: Use the content provided *within* the tag **
    # Content might be empty if LLM failed to put it inside
    if content is None: # Check if content is None (should be string, even empty)
        print(f"Warning: FILEWRITE called for '{path}' but content was None.")
        content = "" # Treat None as empty string

    # Clean potential markdown only after confirming content exists
    clean_content = clean_code_for_file(content)
    print(f"Cleaned content length for '{path}': {len(clean_content)}")
    if not clean_content and content: # Log if cleaning removed everything
         print(f"Warning: Content for '{path}' became empty after cleaning markdown.")

    try:
        # Ensure parent directory exists *before* opening the file
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        with open(abs_path, "w", encoding='utf-8') as f: # Use utf-8
            f.write(clean_content)
        return f"File '{path}' written successfully ({len(clean_content)} bytes)." # Provide feedback on size
    except Exception as e:
        print(f"ERROR in write_file: {traceback.format_exc()}")
        return f"Exception writing file '{path}': {str(e)}"

def read_file(attributes, content):
    path = attributes.get("path")
    print(f"Executing FILEREAD: {path}")
    if not path: return "Error: Missing path attribute"
    
    # Clean any markdown fences from the path
    path = clean_code_for_file(path)

    abs_path, error = _resolve_safe_path(path)
    if error: return error

    try:
        if not os.path.exists(abs_path):
            return f"Error: File not found at '{path}'"
        if not os.path.isfile(abs_path):
             return f"Error: Path '{path}' is not a file."

        # Add size limit for safety?
        MAX_READ_SIZE = 100 * 1024 # 100 KB limit
        file_size = os.path.getsize(abs_path)
        if file_size > MAX_READ_SIZE:
            return f"Error: File '{path}' is too large ({file_size} bytes > {MAX_READ_SIZE} bytes limit)."

        with open(abs_path, "r", encoding='utf-8') as f: # Use utf-8
            file_content = f.read()
        # Return the actual content
        return file_content
    except Exception as e:
        print(f"ERROR in read_file: {traceback.format_exc()}")
        return f"Exception reading file '{path}': {str(e)}"

def web_request(attributes, content):
    url = attributes.get("url")
    print(f"Executing WEBREQUEST: {url}")
    if not url:
        return "Error: Missing url attribute"
    MAX_WEB_SIZE = 400 * 1024
    
    # Try Playwright with anti-detection measures
    try:
        from playwright.sync_api import sync_playwright
        import random
        import time
        
        with sync_playwright() as p:
            # Use more realistic browser options
            browser = p.chromium.launch(
                headless=True,
                args=[
                    '--disable-blink-features=AutomationControlled',
                    '--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36',
                ]
            )
            
            # Create context with more human-like properties
            context = browser.new_context(
                viewport={'width': 1920, 'height': 1080},
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36',
                locale='en-US',
                timezone_id='America/New_York',
                has_touch=False,
            )
            
            # Add human-like behavior
            page = context.new_page()
            
            # Override certain JavaScript properties that are used to detect automation
            page.add_init_script("""
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => false,
                });
                
                // Override the permissions API
                const originalQuery = window.navigator.permissions.query;
                window.navigator.permissions.query = (parameters) => {
                    if (parameters.name === 'notifications') {
                        return Promise.resolve({ state: Notification.permission });
                    }
                    return originalQuery(parameters);
                };
                
                // Add some plugins to seem more human-like
                Object.defineProperty(navigator, 'plugins', {
                    get: () => [
                        { name: 'Chrome PDF Plugin', filename: 'internal-pdf-viewer' },
                        { name: 'Chrome PDF Viewer', filename: 'mhjfbmdgcfjbbpaeojofohoefgiehjai' },
                        { name: 'Native Client', filename: 'internal-nacl-plugin' }
                    ]
                });
            """)
            
            # Simulate more realistic browsing behavior
            page.goto(url, timeout=60000, wait_until='domcontentloaded')
            
            # Add random pauses and movements to appear more human-like
            time.sleep(random.uniform(1, 3))
            
            # Scroll down a bit
            page.mouse.move(random.randint(100, 500), random.randint(100, 300))
            for _ in range(3):
                page.keyboard.press("ArrowDown")
                time.sleep(random.uniform(0.1, 0.3))
            
            # Wait for network to be mostly idle
            page.wait_for_load_state("networkidle", timeout=15000)
            
            # Extract content
            html_content = page.content()
            
            # Clean up
            context.close()
            browser.close()
            
            if len(html_content.encode('utf-8')) > MAX_WEB_SIZE:
                return f"Error: Web response from '{url}' is too large."
            
            # Check if the page likely contains a CAPTCHA
            captcha_indicators = ['captcha', 'robot', 'human verification', 'security check', 'prove you are human']
            if any(indicator.lower() in html_content.lower() for indicator in captcha_indicators):
                print(f"CAPTCHA likely detected for URL: {url}, falling back to requests")
                raise Exception("CAPTCHA detected, falling back to requests")
                
            return html_content
            
    except Exception as e:
        print(f"Playwright failed or CAPTCHA detected, falling back to requests: {e}")
        # Fallback to requests
        try:
            import requests
            
            # Use a more realistic user agent and add more headers
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate, br',
                'Referer': 'https://www.google.com/',
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
                'Sec-Fetch-Dest': 'document',
                'Sec-Fetch-Mode': 'navigate',
                'Sec-Fetch-Site': 'cross-site',
                'Cache-Control': 'max-age=0',
            }
            
            session = requests.Session()
            response = session.get(url, timeout=30, headers=headers)
            response.raise_for_status()
            
            if len(response.content) > MAX_WEB_SIZE:
                return f"Error: Web response from '{url}' is too large ({len(response.content)} bytes > {MAX_WEB_SIZE} bytes)."
            
            try:
                return response.content.decode('utf-8')
            except UnicodeDecodeError:
                return response.text
                
        except requests.exceptions.Timeout:
            return f"Error: Request to '{url}' timed out."
        except requests.exceptions.RequestException as e:
            print(f"ERROR in web_request fallback: {e}")
            error_msg = f"Exception during web request to '{url}': {str(e)}"
            if hasattr(e, 'response') and e.response is not None:
                error_msg += f" (Status Code: {e.response.status_code})"
            return error_msg
        except Exception as e:
            print(f"ERROR in web_request fallback: {e}")
            return f"Unexpected exception during web request to '{url}': {str(e)}"
            
    # Consider adding a third fallback: Using a news API for common queries
    if "news" in url.lower() or any(domain in url.lower() for domain in ["cnn.com", "bbc.com", "reuters.com", "nytimes.com"]):
        try:
            # You could use a news API like NewsAPI, GDELT, or similar here
            print("Web requests failed, could consider using a dedicated news API as another fallback")
            pass  # Implement your news API logic here if needed
        except Exception as e:
            print(f"News API fallback also failed: {e}")
    
    return "Error: All web request methods failed."

def execute_script(attributes, content):
    path = attributes.get("path")
    print(f"Executing EXECUTE: {path}")
    if not path: return "Error: Missing path attribute"
    
    # Clean any markdown fences from the path
    path = clean_code_for_file(path)

    abs_path, error = _resolve_safe_path(path)
    if error: return error

    if not os.path.exists(abs_path):
        return f"Error: Script file not found at '{path}'"
    if not os.path.isfile(abs_path):
        return f"Error: Path '{path}' is not a file."

    try:
        file_extension = os.path.splitext(path)[1].lower()
        cmd = []
        interpreter = None
        interpreter_path = None # Store full path if found

        if file_extension == '.py':
            interpreter = 'python'
        elif file_extension in ['.sh', '.bash']:
            interpreter = 'bash'
        elif file_extension in ['.js']:
             interpreter = 'node'
        # Add other interpreters if needed (e.g., 'php', 'perl')

        # Find interpreter path robustly
        if interpreter:
             import shutil
             interpreter_path = shutil.which(interpreter)
             if not interpreter_path:
                  return f"Error: Interpreter '{interpreter}' not found in system PATH."
             cmd = [interpreter_path, abs_path]
        else: # Try executing directly (requires execute permissions on the file itself)
            if os.access(abs_path, os.X_OK):
                 cmd = [abs_path]
            else:
                return f"Error: File '{path}' is not executable and has unknown extension for interpretation."

        print(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60, cwd=ALLOWED_DIR, check=False)
        output = result.stdout if result.returncode == 0 else f"Error (Return Code {result.returncode}): {result.stderr}"
        return output.strip()

    except subprocess.TimeoutExpired:
        return f"Error: Script execution timed out after 60 seconds."
    except Exception as e:
        print(f"ERROR in execute_script: {traceback.format_exc()}")
        return f"Exception executing script '{path}': {str(e)}"

tag_functions = {
    "TERMINAL": execute_terminal,
    "FILEWRITE": write_file,
    "FILEREAD": read_file,
    "WEBREQUEST": web_request,
    "EXECUTE": execute_script
    # CONFIRM is handled by frontend/main loop, not directly executed here
}

def process_tags_and_get_results(tags):
    """Executes tags and returns a dictionary with results or confirmation info."""
    results = []
    needs_confirmation = False
    confirmation_prompt = None

    for tag_info in tags:
        tag_name = tag_info["tag"] # Already uppercase from parse_tags
        attributes = tag_info["attributes"]
        content = tag_info["content"]
        result_text = f"Error: Unknown tag '{tag_name}'"

        if tag_name == "CONFIRM":
            # Signal frontend - stop processing further tags in this response
            needs_confirmation = True
            confirmation_prompt = content
            print(f"CONFIRM tag found: '{confirmation_prompt}'")
            # Add a placeholder result for the CONFIRM tag itself if needed by history logic
            results.append({
                 "tag": tag_name,
                 "attributes": attributes,
                 "content": content,
                 "result": "Awaiting user confirmation..." # Placeholder result
            })
            break # Stop processing tags after CONFIRM

        elif tag_name in tag_functions:
            func = tag_functions[tag_name]
            try:
                result_text = func(attributes, content)
            except Exception as e:
                print(f"ERROR during tag execution ({tag_name}): {traceback.format_exc()}")
                result_text = f"Exception during {tag_name} execution: {str(e)}"
            results.append({
                "tag": tag_name,
                "attributes": attributes,
                "content": content, # Keep original content for context
                "result": result_text
            })
        else:
             # Handle unknown tags if necessary, or just append the default error
             print(f"Warning: Encountered unknown tag '{tag_name}'")
             results.append({
                "tag": tag_name,
                "attributes": attributes,
                "content": content,
                "result": result_text # The "Unknown tag" error
             })


    return {
        "action_results": results,
        "needs_confirmation": needs_confirmation,
        "confirmation_prompt": confirmation_prompt
    }


# --- Message Builders (Using specific _RESULT tags) ---

def _format_result_for_llm(tag, result):
    """Formats the action result for LLM consumption."""
    # Escape the result content to prevent issues if it contains XML-like chars
    escaped_result = html.escape(str(result))
    return f"<{tag}_RESULT>{escaped_result}</{tag}_RESULT>"

def build_openai_messages(history):
    print("Building OpenAI messages...")
    # Persona description can be added here
    system_message = TAG_INFO + "\nYou are a helpful AI assistant."
    messages = [{"role": "system", "content": system_message}]

    for entry in history:
        entry_type = entry.get("type")
        content = entry.get("content")
        role = None

        if entry_type == "user":
            role = "user"
        elif entry_type == "llm":
            role = "assistant"
            # ** Important: Filter out tool tags from assistant's *past* messages
            #    if they were successfully processed in the *next* action turn.
            #    This prevents the LLM seeing its own tags that were already acted upon.
            #    However, this requires more complex history management.
            #    For now, we leave them, but structure the _RESULT tags clearly.
            # content = TAG_PATTERN.sub('', content).strip() # Basic cleaning (might be too aggressive)
        elif entry_type == "action":
            # Action results are presented back to the LLM as user input
            role = "user"
            tag = entry.get('tag', 'UNKNOWN_TAG').upper()
            result = entry.get('result', 'Error: Missing result in history')
            # Format using the specific _RESULT tag
            content = _format_result_for_llm(tag, result)

        # Append if valid
        if role and content is not None:
            # Basic check for meaningful content
            if isinstance(content, str) and len(content.strip()) == 0:
                 print(f"Skipping empty message for role {role} in history.")
                 continue
            messages.append({"role": role, "content": content})
        elif role: # Log if content is None but role is set
            print(f"Warning: History entry has role '{role}' but None content. Type: '{entry_type}'. Entry: {entry}")

    # The caller (`chat_endpoint`) will append the latest user message/confirmation result
    print(f"OpenAI messages built (count: {len(messages)}): {messages[-3:]}") # Log last few messages
    return messages


def build_gemini_prompt(history):
    print("Building Gemini prompt...")
    # Persona description
    prompt = "System: " + TAG_INFO + "\nYou are a helpful AI assistant.\n\n"

    for entry in history:
        entry_type = entry.get("type")
        content = entry.get("content")

        if entry_type == "user":
            prompt += f"Human: {content}\n\n"
        elif entry_type == "llm":
            # See comment in build_openai_messages about potentially cleaning tags
            prompt += f"Assistant: {content}\n\n"
        elif entry_type == "action":
            tag = entry.get('tag', 'UNKNOWN_TAG').upper()
            result = entry.get('result', 'Error: Missing result in history')
            # Format using the specific _RESULT tag, presented as Human input
            result_tag_content = _format_result_for_llm(tag, result)
            prompt += f"Human: {result_tag_content}\n\n"
        elif content is None:
             print(f"Warning: History entry type '{entry_type}' has None content. Entry: {entry}")

    # Caller appends final "Human: ..." and "Assistant:"
    print(f"Gemini prompt built:\n...{prompt[-200:]}") # Log end of prompt
    return prompt

def build_claude_prompt(history):
    print("Building Claude messages...")
    # Persona description
    system_message = TAG_INFO + "\nYou are a helpful AI assistant."
    messages = []

    for entry in history:
        role = None
        content = entry.get("content")
        entry_type = entry.get("type")

        if entry_type == "user":
            role = "user"
        elif entry_type == "llm":
            role = "assistant"
            # See comment in build_openai_messages about potentially cleaning tags
        elif entry_type == "action":
            role = "user" # Treat action results as user feedback
            tag = entry.get('tag', 'UNKNOWN_TAG').upper()
            result = entry.get('result', 'Error: Missing result in history')
            # Format using the specific _RESULT tag
            content = _format_result_for_llm(tag, result)

        if role and content is not None:
            if isinstance(content, str) and len(content.strip()) == 0:
                 print(f"Skipping empty message for role {role} in history.")
                 continue
            messages.append({"role": role, "content": content})
        elif role:
             print(f"Warning: History entry type '{entry_type}' role '{role}' has None content. Entry: {entry}")

    # Structure for Claude API - Caller adds the final user message
    claude_data = {
        "model": "claude-3-haiku-20240307", # Use Haiku for speed/cost, or Sonnet/Opus
        "max_tokens": 4000,
        "system": system_message,
        "messages": messages
    }
    print(f"Claude messages built (count: {len(messages)}): {messages[-3:]}") # Log last few
    return claude_data

message_builders = {
    "openai": build_openai_messages,
    "gemini": build_gemini_prompt,
    "claude": build_claude_prompt
}

# --- API Callers (Using official clients where possible) ---

def openai_api_call(messages):
    print("Calling OpenAI API...")
    api_key = current_api_keys.get("openai")
    if not api_key: return "Error: OpenAI API Key not set."
    try:
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o", # Use a capable model like gpt-4o or gpt-4-turbo
            messages=messages,
            temperature=0.6, # Adjust temperature if needed
        )
        return response.choices[0].message.content
    except openai.APIConnectionError as e:
        print(f"ERROR: OpenAI API connection error: {e}")
        return f"Error (OpenAI): Connection error - {e}"
    except openai.RateLimitError as e:
        print(f"ERROR: OpenAI API rate limit exceeded: {e}")
        return f"Error (OpenAI): Rate limit exceeded - {e}"
    except openai.APIStatusError as e:
        print(f"ERROR: OpenAI API status error: {e.status_code} - {e.response}")
        return f"Error (OpenAI): API error {e.status_code} - {e.message}"
    except Exception as e:
        print(f"ERROR: OpenAI API call failed: {traceback.format_exc()}")
        return f"Error (OpenAI): Unexpected error - {str(e)}"

def configure_gemini_if_needed():
    """Configures Gemini API if necessary using current_api_keys."""
    gemini_key = current_api_keys.get("gemini")
    if not gemini_key and GEMINI_API_KEY_HARDCODED:
         print("Warning: Using hardcoded Gemini API key.")
         gemini_key = GEMINI_API_KEY_HARDCODED

    if not gemini_key:
        print("Error: Gemini API Key is not available (checked UI, env var, and hardcoded).")
        return False

    # Check if already configured with the *same* key to avoid redundant calls
    try:
        # Accessing internal config is not ideal, but avoids reconfiguring unnecessarily.
        # A better approach might be a global flag or checking `genai.api_key` if stable.
        # For simplicity, we'll just reconfigure if the key might have changed.
        # Let's assume we need to configure each time for safety unless we track the configured key.
        print(f"Configuring Gemini with key ending ...{gemini_key[-4:]}")
        genai.configure(api_key=gemini_key)
        # Test model instantiation
        genai.GenerativeModel("gemini-1.5-flash") # Or "gemini-pro"
        print("Gemini configured successfully.")
        return True
    except Exception as e:
        print(f"ERROR: Failed to configure Gemini: {traceback.format_exc()}")
        return False


def gemini_api_call(prompt):
    print("Calling Gemini API...")
    if not configure_gemini_if_needed():
        return "Error: Gemini API Key not configured or invalid."

    try:
        # Use safety settings to be less restrictive if needed, but be careful
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ]
        model = genai.GenerativeModel("gemini-1.5-flash", safety_settings=safety_settings)
        response = model.generate_content(prompt)

        # More robust response checking
        if response.parts:
            return response.text.strip()
        elif hasattr(response, 'candidates') and response.candidates:
             candidate = response.candidates[0]
             if hasattr(candidate, 'content') and candidate.content.parts:
                  return candidate.content.text.strip()
             else:
                  # Check finish reason and safety ratings for blocked content
                  finish_reason = getattr(candidate, 'finish_reason', 'UNKNOWN')
                  safety_ratings = getattr(candidate, 'safety_ratings', [])
                  err_msg = f"Gemini response finished with reason: {finish_reason}."
                  if safety_ratings: err_msg += f" Safety Ratings: {safety_ratings}."
                  # Also check prompt feedback if available
                  block_reason = getattr(response, 'prompt_feedback', {}).get('block_reason', None)
                  if block_reason: err_msg += f" Prompt Feedback Block Reason: {block_reason}."
                  print(f"ERROR: Gemini returned empty/blocked content: {err_msg}")
                  return f"Error (Gemini): {err_msg}"
        else:
             # Unexpected response structure
             print(f"ERROR: Unexpected Gemini response structure: {response}")
             return f"Error (Gemini): Unexpected response format."

    except Exception as e:
        print(f"ERROR: Gemini API call failed: {traceback.format_exc()}")
        error_detail = str(e)
        # Add specific Google API error details if available
        if hasattr(e, 'message'): error_detail = e.message
        if hasattr(e, 'reason'): error_detail += f" (Reason: {e.reason})"
        return f"Error (Gemini): {error_detail}"


def claude_api_call(data):
    print("Calling Claude API...")
    api_key = current_api_keys.get("claude")
    if not api_key: return "Error: Anthropic API Key not set (check ANTHROPIC_API_KEY env var or UI input)."

    try:
        # Use the official client
        client = anthropic.Anthropic(api_key=api_key) # Client reads ANTHROPIC_API_KEY env var by default if not passed
        response = client.messages.create(
            model=data["model"],
            system=data["system"],
            messages=data["messages"],
            max_tokens=data["max_tokens"]
        )

        # Extract text content
        if response.content and isinstance(response.content, list) and len(response.content) > 0:
            # Find the first text block, as recommended by Anthropic docs
            for block in response.content:
                if block.type == 'text':
                    return block.text
            print("ERROR: Claude response content found, but no 'text' block.")
            return "Error (Claude): Response received but no text content found."
        else:
             print(f"ERROR: Unexpected Claude response content structure: {response.content}")
             # Check stop reason
             stop_reason = response.stop_reason
             return f"Error (Claude): Empty or unexpected content. Stop Reason: {stop_reason}"

    except anthropic.APIConnectionError as e:
        print(f"ERROR: Claude API connection error: {e}")
        return f"Error (Claude): Connection error - {e}"
    except anthropic.RateLimitError as e:
        print(f"ERROR: Claude API rate limit exceeded: {e}")
        return f"Error (Claude): Rate limit exceeded - {e}"
    except anthropic.APIStatusError as e:
        print(f"ERROR: Claude API status error: {e.status_code} - {e.response}")
        # Try to extract message from response if possible
        error_message = str(e)
        try:
            error_details = e.response.json()
            error_message = error_details.get('error', {}).get('message', str(e))
        except: pass # Ignore parsing errors
        return f"Error (Claude): API error {e.status_code} - {error_message}"
    except Exception as e:
        print(f"ERROR: Claude API call failed: {traceback.format_exc()}")
        return f"Error (Claude): Unexpected error - {str(e)}"


api_callers = {
    "openai": openai_api_call,
    "gemini": gemini_api_call,
    "claude": claude_api_call
}

# --- Flask Route ---

@app.route('/chat', methods=['POST'])
def chat_endpoint():
    print("\n--- Received /chat request ---")
    data = request.json
    if not data:
        return jsonify({"error": "Invalid JSON payload"}), 400

    provider = data.get('provider')
    history_from_frontend = data.get('history', [])
    prompt = data.get('prompt') # New user message
    confirmation_result = data.get('confirmation_result') # 'Approved' or 'Rejected'
    ui_api_keys = data.get('api_keys', {})
    
    # Track if we need to send action results to LLM
    previous_action_results = data.get('action_results', [])

    # --- Update API Keys ---
    # Prioritize UI keys, then env vars
    with history_lock: # Protect access if multiple threads handle requests
        current_api_keys["openai"] = ui_api_keys.get("openai") or os.getenv("OPENAI_API_KEY")
        current_api_keys["gemini"] = ui_api_keys.get("gemini") or os.getenv("GEMINI_API_KEY")
        current_api_keys["claude"] = ui_api_keys.get("claude") or os.getenv("ANTHROPIC_API_KEY") # Use standard env var
    print(f"Using Provider: {provider}")
    print(f"API Keys Set: OpenAI: {'Yes' if current_api_keys['openai'] else 'No'}, Gemini: {'Yes' if current_api_keys['gemini'] else 'No'}, Claude: {'Yes' if current_api_keys['claude'] else 'No'}")

    # --- Select Builder and Caller ---
    builder = message_builders.get(provider)
    api_caller = api_callers.get(provider)
    if not builder or not api_caller:
        print(f"Error: Unsupported provider '{provider}'")
        return jsonify({"error": f"Unsupported provider: {provider}"}), 400

    # --- History & Input Preparation ---
    current_turn_history = list(history_from_frontend) # Use state from frontend
    llm_context_input = None # Data structure for the specific API
    latest_user_turn_content = None # The final content for the 'user' turn

    if confirmation_result:
        print(f"Processing confirmation result: {confirmation_result}")
        # Format the confirmation result as the user's input for *this* turn
        latest_user_turn_content = _format_result_for_llm("CONFIRM", confirmation_result)

        # Update the corresponding action in history (frontend should already have done this)
        # Find the *last* action entry in history that was a CONFIRM tag *without* a result yet.
        found_and_updated = False
        for i in range(len(current_turn_history) - 1, -1, -1):
             entry = current_turn_history[i]
             if entry.get("type") == "action" and entry.get("tag") == "CONFIRM":
                  # Check if result is missing or was the placeholder
                  if entry.get("result") is None or entry.get("result") == "Awaiting user confirmation...":
                       entry["result"] = confirmation_result # Update in place
                       print(f"Updated history action at index {i} with CONFIRM result.")
                       found_and_updated = True
                       break
                  elif entry.get("result") == confirmation_result:
                       # Already updated, likely by frontend - this is fine.
                       found_and_updated = True # Treat as updated
                       break
        if not found_and_updated:
             print("Warning: Did not find matching CONFIRM action in history to update, or it was already updated differently.")
             # Proceed anyway, relying on the history sent by frontend.

        # Build context from the updated history
        llm_context_input = builder(current_turn_history)

    elif prompt:
        print(f"Processing new prompt (first 100 chars): {prompt[:100]}")
        latest_user_turn_content = prompt
        # Add the raw prompt to history *before* building context for it
        # Note: builder *doesn't* modify the list, it just reads it
        current_turn_history.append({"type": "user", "content": prompt})
        # Build context based on the history that *includes* this new prompt
        llm_context_input = builder(current_turn_history)

    else:
        print("Error: Request missing prompt or confirmation_result")
        return jsonify({"error": "Request must contain either a prompt or a confirmation_result"}), 400

    # --- Add the latest user turn content to the API-specific format ---
    if not latest_user_turn_content:
         print("Error: latest_user_turn_content is unexpectedly empty. Cannot call LLM.")
         return jsonify({"error": "Internal error preparing user input"}), 500

    if provider == 'openai':
        # Append the final user message object
        llm_context_input.append({"role": "user", "content": latest_user_turn_content})
    elif provider == 'gemini':
        # Append the final Human turn and the Assistant prompt marker
        llm_context_input += f"Human: {latest_user_turn_content}\n\nAssistant:"
    elif provider == 'claude':
        # Append the final user message object
        llm_context_input['messages'].append({"role": "user", "content": latest_user_turn_content})

    # --- Call LLM API ---
    print(f"Calling {provider} API...")
    llm_response_text = api_caller(llm_context_input)
    print(f"LLM Response received (first 100 chars): {llm_response_text[:100]}...")

    # Check for API errors signaled by the caller function
    if not llm_response_text or (isinstance(llm_response_text, str) and llm_response_text.startswith("Error:")):
        print(f"LLM API Error: {llm_response_text}")
        # Return the error directly in the response payload
        return jsonify({
            "llm_response": llm_response_text,
            "action_results": [],
            "needs_confirmation": False
        }), 200 # Return 200 OK, but payload indicates the error    # --- Process Tags in LLM Response ---
    print("Parsing tags in LLM response...")
    tags = parse_tags(llm_response_text)
    action_processing_result = {"action_results": []} # Default

    if tags:
        print(f"Found tags: {[t['tag'] for t in tags]}")
        # Execute tags or check for confirmation
        action_processing_result = process_tags_and_get_results(tags)
        print(f"Action processing result: {action_processing_result}")
        
        # If we have action results, we should send them to the LLM for processing
        if action_processing_result["action_results"] and not action_processing_result["needs_confirmation"]:
            # Format each action result and add to history
            action_inputs = []
            for result in action_processing_result["action_results"]:
                tag = result["tag"]
                result_text = result["result"]
                action_inputs.append(_format_result_for_llm(tag, result_text))
            
            # Add action results to history
            system_message = "System: Here are the results of the actions you requested:\n\n" + "\n".join(action_inputs)
            current_turn_history.append({"type": "user", "content": system_message})
            
            # Rebuild context with action results
            llm_context_input = builder(current_turn_history)
            if provider == 'openai':
                llm_context_input.append({"role": "user", "content": system_message})
            elif provider == 'gemini':
                llm_context_input += f"Human: {system_message}\n\nAssistant:"
            elif provider == 'claude':
                llm_context_input['messages'].append({"role": "user", "content": system_message})

            # Call LLM again to process results
            print("Calling LLM to process action results...")
            llm_response_text = api_caller(llm_context_input)
            print(f"LLM Response to action results (first 100 chars): {llm_response_text[:100]}...")
            
    else:
        print("No tags found in LLM response.")

    # --- Prepare Response for Frontend ---
    # Decide if you want to send the LLM response with tags removed or as is.
    # Sending raw allows frontend to potentially style tags differently.
    final_llm_response_for_frontend = llm_response_text

    # Optional: Clean the response text *after* tags are processed
    # if action_processing_result.get("action_results") or action_processing_result.get("needs_confirmation"):
    #      # Example: Simple removal (might leave awkward phrasing)
    #      cleaned_text = TAG_PATTERN.sub('', llm_response_text).strip()
    #      # More sophisticated cleaning might be needed depending on LLM behavior
    #      final_llm_response_for_frontend = cleaned_text

    response_payload = {
        "llm_response": final_llm_response_for_frontend, # Send raw or cleaned
        "action_results": action_processing_result.get("action_results", []),
        "needs_confirmation": action_processing_result.get("needs_confirmation", False),
        "confirmation_prompt": action_processing_result.get("confirmation_prompt", None)
    }

    print(f"Sending response to frontend: Needs Confirm: {response_payload['needs_confirmation']}, Actions: {len(response_payload['action_results'])}")
    return jsonify(response_payload)

# --- Main Execution ---
if __name__ == '__main__':
    print("--- Starting Flask Server ---")
    # Use debug=True for development (enables auto-reload and detailed errors)
    # Use debug=False for production!
    # host='0.0.0.0' makes it accessible on your network
    app.run(host='127.0.0.1', port=5000, debug=True)