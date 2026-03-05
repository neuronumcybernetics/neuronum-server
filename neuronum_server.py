import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import asyncio
import sys
import platform
import textwrap
from datetime import datetime
from openai import OpenAI
import aiosqlite
from typing import List, Tuple
from neuronum import Cell
import logging
import subprocess
import json
from jinja2 import Environment, FileSystemLoader

import tool_registry
from config import (
    LOG_FILE,
    DB_PATH,
    MODEL_MAX_TOKENS,
    MODEL_TEMPERATURE,
    MODEL_TOP_P,
    VLLM_MODEL_NAME,
    VLLM_API_BASE,
    OLLAMA_MODEL_NAME,
    OLLAMA_API_BASE,
    CONVERSATION_HISTORY_LIMIT,
    TEMPLATES_DIR
)

# Detect backend: Apple Silicon → Ollama, NVIDIA GPU → vLLM
def detect_backend():
    """Detect hardware and return (api_base, model_name, backend_name)"""
    is_mac_arm = platform.system() == "Darwin" and platform.machine() == "arm64"
    if is_mac_arm:
        return OLLAMA_API_BASE, OLLAMA_MODEL_NAME, "ollama"
    else:
        return VLLM_API_BASE, VLLM_MODEL_NAME, "vllm"

API_BASE, MODEL_NAME, BACKEND = detect_backend()

# Setup Jinja2 environment
env = Environment(loader=FileSystemLoader(TEMPLATES_DIR), autoescape=True)

# Logging Setup
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler(LOG_FILE, mode='a')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Database Functions

async def enable_wal(db_path=DB_PATH):
    """Enable WAL mode for better database concurrency"""
    async with aiosqlite.connect(db_path) as db:
        await db.execute("PRAGMA journal_mode=WAL;")
        await db.commit()
        logging.info("WAL mode enabled.")

async def init_db(db_path=DB_PATH):
    """Initialize database tables"""
    async with aiosqlite.connect(db_path) as db:
        await db.execute('''
            CREATE TABLE IF NOT EXISTS memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user TEXT,
                role TEXT,
                message TEXT,
                message_type TEXT DEFAULT 'chat',
                context TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        await db.execute('''
            CREATE TABLE IF NOT EXISTS actions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                operator TEXT,
                subject TEXT,
                description TEXT,
                original_data TEXT,
                tool_id TEXT,
                tool_name TEXT,
                parameter TEXT,
                response TEXT,
                status TEXT DEFAULT 'pending',
                is_multi_step BOOLEAN,
                steps TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        await db.commit()
        logging.info("Database initialized.")

async def store_message(user, role, message, message_type="chat", context=None, db_path=DB_PATH):
    """Store message to memory table for conversation history

    Args:
        user: User identifier
        role: Message role (user, assistant, system)
        message: The message content
        message_type: Type of message (chat, action_result, system_notification)
        context: Optional additional context (e.g., action_id reference)
    """
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "INSERT INTO memory (user, role, message, message_type, context) VALUES (?, ?, ?, ?, ?)",
            (user, role, message, message_type, context)
        )
        await db.commit()

async def store_action_entry(operator: str, subject: str, context: str, original_data: str, tool_id: str = None, tool_name: str = None, parameter: str = None, is_multi_step: bool = False, steps: str = None, status: str = 'pending', response: str = None, db_path=DB_PATH) -> int:
    """Store action entry in actions table

    Args:
        operator: The cell ID of the user who triggered the action
        status: Action status ('pending', 'finished', 'failed', 'dismissed')
        response: Only used for failed actions to store error message

    Returns:
        The ID of the newly created action entry
    """
    async with aiosqlite.connect(db_path) as db:
        cursor = await db.execute(
            "INSERT INTO actions (operator, subject, description, original_data, tool_id, tool_name, parameter, status, response, is_multi_step, steps) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (operator, subject, context, original_data, tool_id, tool_name, parameter, status, response, is_multi_step, steps)
        )
        action_id = cursor.lastrowid
        await db.commit()
        if is_multi_step:
            logging.info(f"Multi-step action entry stored: {subject} (ID: {action_id}, status: {status})")
        else:
            logging.info(f"Single-step action entry stored: {subject} (ID: {action_id}, status: {status})")
        return action_id

async def fetch_pending_action(action_id: int, db_path=DB_PATH) -> dict | None:
    """Fetch a pending action by ID for approval/execution"""
    async with aiosqlite.connect(db_path) as db:
        query = """SELECT id, subject, description, original_data, tool_id, tool_name, parameter, status, is_multi_step, steps
                   FROM actions WHERE id = ? AND status = 'pending'"""
        async with db.execute(query, (action_id,)) as cursor:
            row = await cursor.fetchone()
            if row:
                return {
                    "id": row[0],
                    "subject": row[1],
                    "description": row[2],
                    "original_data": row[3],
                    "tool_id": row[4],
                    "tool_name": row[5],
                    "parameter": row[6],
                    "status": row[7],
                    "is_multi_step": bool(row[8]),
                    "steps": row[9]
                }
            return None

async def update_action_status(action_id: int, status: str, response: str = None, db_path=DB_PATH):
    """Update an action's status and optionally its response"""
    async with aiosqlite.connect(db_path) as db:
        if response:
            await db.execute(
                "UPDATE actions SET status = ?, response = ? WHERE id = ?",
                (status, response, action_id)
            )
        else:
            await db.execute(
                "UPDATE actions SET status = ? WHERE id = ?",
                (status, action_id)
            )
        await db.commit()
        logging.info(f"Action {action_id} status updated to: {status}")

async def fetch_latest_messages(user, limit=CONVERSATION_HISTORY_LIMIT, db_path=DB_PATH) -> List[Tuple[str, str]]:
    """Fetch latest N messages for conversation history

    Args:
        user: User identifier
        limit: Maximum number of messages to fetch
    """
    async with aiosqlite.connect(db_path) as db:
        query = "SELECT role, message FROM memory WHERE user = ? ORDER BY id DESC LIMIT ?"
        params = (user, limit)

        async with db.execute(query, params) as cursor:
            rows = await cursor.fetchall()
            return list(reversed(rows))

def validate_tool_parameters(parameters: dict, input_schema: dict) -> tuple[bool, str]:
    """Validate parameters against JSON schema and return (is_valid, error_message) tuple"""
    if not input_schema:
        return True, ""

    properties = input_schema.get("properties", {})
    required = input_schema.get("required", [])

    for req_param in required:
        if req_param not in parameters:
            return False, f"Missing required parameter: '{req_param}'"

    for param_name, param_value in parameters.items():
        if param_name not in properties:
            continue

        param_schema = properties[param_name]
        expected_type = param_schema.get("type", "string")

        type_map = {
            "string": str,
            "number": (int, float),
            "integer": int,
            "boolean": bool,
            "array": list,
            "object": dict
        }

        expected_python_type = type_map.get(expected_type)
        if expected_python_type and not isinstance(param_value, expected_python_type):
            return False, (
                f"Parameter '{param_name}' has wrong type. "
                f"Expected {expected_type}, got {type(param_value).__name__}"
            )

        if "enum" in param_schema:
            if param_value not in param_schema["enum"]:
                return False, (
                    f"Parameter '{param_name}' value '{param_value}' not in allowed values: "
                    f"{param_schema['enum']}"
                )

        if expected_type == "array" and "items" in param_schema:
            item_type = param_schema["items"].get("type")
            if item_type and item_type in type_map:
                item_python_type = type_map[item_type]
                for idx, item in enumerate(param_value):
                    if not isinstance(item, item_python_type):
                        return False, (
                            f"Parameter '{param_name}' array item at index {idx} has wrong type. "
                            f"Expected {item_type}, got {type(item).__name__}"
                        )

    return True, ""


# OpenAI Client Configuration (works with both vLLM and Ollama)
try:
    logging.info(f"Backend: {BACKEND} | Connecting to {API_BASE} | Model: {MODEL_NAME}")

    client = OpenAI(
        base_url=API_BASE,
        api_key="EMPTY"
    )

    logging.info(f"OpenAI client initialized for {BACKEND} server")

except Exception as e:
    logging.error(f"Error initializing OpenAI client: {e}")
    if BACKEND == "ollama":
        logging.error("Make sure Ollama is running: ollama serve")
    else:
        logging.error("Make sure vLLM is running: python -m vllm.entrypoints.openai.api_server --model <model-name>")
    sys.exit(1)

def create_chat_completion(messages, max_tokens=MODEL_MAX_TOKENS, temperature=MODEL_TEMPERATURE):
    """Generate chat completion using OpenAI-compatible API (vLLM or Ollama)"""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=MODEL_TOP_P,
        )

        return {
            'choices': [
                {
                    'message': {
                        'content': response.choices[0].message.content
                    }
                }
            ]
        }

    except Exception as e:
        logging.error(f"Error in create_chat_completion: {e}")
        if BACKEND == "ollama":
            logging.error("Make sure Ollama is running: ollama serve")
        else:
            logging.error("Make sure vLLM is running at http://127.0.0.1:8000")
        raise

async def fetch_all_actions(operator: str = None, db_path=DB_PATH) -> List[dict]:
    """Fetch action entries from actions table (audit log), optionally filtered by operator"""
    async with aiosqlite.connect(db_path) as db:
        if operator:
            query = """SELECT id, operator, subject, description, original_data, tool_id, tool_name, parameter, response, status, is_multi_step, steps, timestamp
                       FROM actions WHERE operator = ? ORDER BY timestamp DESC"""
            async with db.execute(query, (operator,)) as cursor:
                rows = await cursor.fetchall()
        else:
            query = """SELECT id, operator, subject, description, original_data, tool_id, tool_name, parameter, response, status, is_multi_step, steps, timestamp
                       FROM actions ORDER BY timestamp DESC"""
            async with db.execute(query) as cursor:
                rows = await cursor.fetchall()

        actions_list = []
        for row in rows:
            action_dict = {
                "id": row[0],
                "subject": row[2],
                "description": row[3],
                "original_data": row[4],
                "tool_id": row[5],
                "tool_name": row[6],
                "parameter": row[7],
                "response": row[8],
                "status": row[9],
                "is_multi_step": bool(row[10]),
                "timestamp": row[12]
            }

            # Parse steps if it's a multi-step action
            if row[10]:  # is_multi_step
                try:
                    action_dict["steps"] = json.loads(row[11]) if row[11] else []
                except json.JSONDecodeError:
                    action_dict["steps"] = []
            else:
                action_dict["steps"] = None

            actions_list.append(action_dict)

        return actions_list

async def erase_data(db_path=DB_PATH):
    """Erase all data from database and clear log file"""
    try:
        async with aiosqlite.connect(db_path) as db:
            await db.execute("DELETE FROM memory")
            logging.info("All conversation history deleted from memory table.")

            await db.commit()

        with open(LOG_FILE, 'w') as f:
            f.write('')
        logging.info("Log file cleared.")

        logging.info("All data erased successfully.")
        return True

    except Exception as e:
        logging.error(f"Error during data erasure: {e}")
        return False


async def convert_tool_result_to_natural_language(
    tool_name: str,
    tool_result: dict,
    subject: str = ""
) -> str:
    """Convert tool JSON response to natural language using LLM (async version)"""
    loop = asyncio.get_running_loop()

    def generate_response():
        try:
            # Extract the actual response content from the tool result
            content_items = tool_result.get("content", [])
            if content_items and len(content_items) > 0:
                tool_output = content_items[0].get("text", str(tool_result))
            else:
                tool_output = str(tool_result)

            # Create prompt for LLM to interpret the response
            prompt = f"""Summarize this tool result in 2-3 short sentences. Focus on the most important highlights only.

Tool: {tool_name}
Task: {subject if subject else 'Tool execution'}

Tool Response:
{tool_output}

Guidelines:
- Maximum 2-3 sentences, be very brief
- Mention only the top 2-3 most important numbers or items
- Do NOT list every item — just summarize the key takeaway
- Start directly with the answer — do NOT start with "Here is", "The result shows", "Summary:", or any preamble
- Write as if you are the AI agent speaking to the user directly
- No praise, no emotional language

Example good responses:
- "Total value is €1.1M across 8 items. Success rate at 75% with €411K completed."
- "3 open tickets, 1 critical. SLA breached on 2 items."
- "Found 5 results matching the query. Top match: Product Overview (last updated Feb 2026)."

Answer:"""

            messages = [{"role": "user", "content": prompt}]

            response = create_chat_completion(
                messages=messages,
                max_tokens=200,
                temperature=0.3
            )

            natural_response = response['choices'][0]['message']['content'].strip()
            return natural_response

        except Exception as e:
            logging.error(f"Error generating natural language response: {e}")
            # Fallback to a simple extraction
            content_items = tool_result.get("content", [])
            if content_items and len(content_items) > 0:
                return content_items[0].get("text", "Tool executed.")
            return "Tool executed."

    return await loop.run_in_executor(None, generate_response)


async def generate_multi_step_summary(subject: str, step_responses: list) -> str:
    """Generate a natural language summary of a multi-step action execution using LLM"""
    loop = asyncio.get_running_loop()

    def generate_summary():
        try:
            steps_text = ""
            for step in step_responses:
                status = step.get("status", "unknown")
                steps_text += f"\nStep {step['step_order']} ({step['step_subject']}): [{status}] {step['response']}"

            prompt = f"""Summarize this multi-step action execution into a brief factual statement.

Task: {subject}

Steps executed:
{steps_text}

Guidelines:
- Write in third person or passive voice - NO "you", "I", "we"
- State facts directly and concisely
- Include specific numbers, amounts, and data from the results
- No congratulations, praise, or emotional language
- If a step failed, clearly state which step failed and why
- 2-3 short sentences maximum

Summary:"""

            messages = [{"role": "user", "content": prompt}]

            response = create_chat_completion(
                messages=messages,
                max_tokens=200,
                temperature=0.3
            )

            return response['choices'][0]['message']['content'].strip()

        except Exception as e:
            logging.error(f"Error generating multi-step summary: {e}")
            success_count = sum(1 for s in step_responses if s.get("status") == "success")
            total = len(step_responses)
            return f"{success_count}/{total} steps completed for: {subject}"

    return await loop.run_in_executor(None, generate_summary)


# Infrastructure Setup

async def setup_infrastructure():
    """Initialize infrastructure for agent"""
    logging.info("Initializing system...")

async def initialize_database():
    """Initialize SQLite database with WAL mode"""
    await enable_wal()
    await init_db()

async def install_tool_requirements():
    """Install requirements from all .config files in tools directory"""
    try:
        in_venv = hasattr(sys, 'real_prefix') or (
            hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
        )

        if not in_venv:
            logging.warning("Warning: Not running in a virtual environment. Skipping automatic package installation.")
            logging.warning("Warning: Please activate a virtual environment and manually install required packages.")
            return

        tools_dir = "./tools"

        if not os.path.exists(tools_dir):
            logging.info("No tools directory found, skipping requirements installation")
            return

        all_requirements = set()

        for filename in os.listdir(tools_dir):
            if filename.endswith(".config"):
                config_path = os.path.join(tools_dir, filename)

                try:
                    with open(config_path, 'r') as f:
                        config_content = f.read()
                        config_data = json.loads(config_content)

                    requirements = config_data.get("requirements", [])

                    if requirements:
                        for req in requirements:
                            all_requirements.add(req)
                        logging.info(f"Found {len(requirements)} requirement(s) in {filename}")

                except json.JSONDecodeError as e:
                    logging.warning(f"Failed to parse {filename}: {e}")
                except Exception as e:
                    logging.warning(f"Error reading {filename}: {e}")

        if not all_requirements:
            logging.info("No tool requirements found in config files")
            return

        logging.info(f"Installing {len(all_requirements)} package(s): {', '.join(all_requirements)}")

        for requirement in all_requirements:
            try:
                logging.info(f"Installing {requirement}...")
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", requirement],
                    capture_output=True,
                    text=True,
                    timeout=120
                )

                if result.returncode == 0:
                    logging.info(f"Successfully installed {requirement}")
                else:
                    logging.error(f"Error: Failed to install {requirement}: {result.stderr}")

            except subprocess.TimeoutExpired:
                logging.error(f"Error: Timeout while installing {requirement}")
            except Exception as e:
                logging.error(f"Error: Error installing {requirement}: {e}")

        logging.info("Tool requirements installation complete")

    except Exception as e:
        logging.error(f"Error in install_tool_requirements: {e}")
        import traceback
        logging.error(traceback.format_exc())


async def setup_cell_connection():
    """Establish connection as Neuronum Cell and return cell instance"""
    cell = Cell()

    if not cell.env.get("HOST"):
        logging.error("Error: No HOST found in Cell credentials. Please run 'neuronum create-cell' or 'neuronum connect-cell' first.")
        await cell.close()
        sys.exit(1)

    logging.info(f"Connected to Cell: {cell.env.get('HOST')}")
    return cell

# Message Handlers

async def send_cell_response(cell, transmitter_id: str, data: dict, public_key: str):
    """Send response back through cell"""
    await cell.tx_response(
        transmitter_id=transmitter_id,
        data=data,
        client_public_key_str=public_key
    )

async def handle_get_status(cell, transmitter: dict):
    """Handle agent status request (health check)"""
    data = transmitter.get("data", {})
    logging.info("Checking Agent Status")

    await send_cell_response(
        cell,
        transmitter.get("transmitter_id"),
        {"json": "running"},
        data.get("public_key", "")
    )

async def handle_get_index(cell, transmitter: dict):
    """Handle getting the index/welcome page for operators"""
    data = transmitter.get("data", {})
    logging.info("Fetching index page")

    index_message = f"Welcome to kybercell - Your Private AI Workspace!"

    # Load and render template
    template = env.get_template("index.html")
    html_content = template.render()

    await send_cell_response(
        cell,
        transmitter.get("transmitter_id"),
        {"json": index_message, "html": html_content},
        data.get("public_key", "")
    )

async def handle_get_actions(cell, transmitter: dict):
    """Handle fetching actions from database (audit log) for the requesting operator"""
    data = transmitter.get("data", {})
    operator = transmitter.get("operator", "")
    logging.info(f"Fetching actions audit log for operator: {operator}")

    actions_list = await fetch_all_actions(operator=operator)
    logging.info(f"Retrieved {len(actions_list)} actions")

    await send_cell_response(
        cell,
        transmitter.get("transmitter_id"),
        {"json": actions_list},
        data.get("public_key", "")
    )


async def handle_approve(cell, transmitter: dict):
    """Handle operator approval of a pending action

    Payload: {type: "approve", action_id: X}
    """
    data = transmitter.get("data", {})
    action_id = data.get("action_id")
    operator = transmitter.get("operator", "anonymous_operator")

    logging.info(f"[Operator {operator}] Approving action {action_id}")

    if not action_id:
        await send_cell_response(
            cell,
            transmitter.get("transmitter_id"),
            {"json": "Error: action_id is required"},
            data.get("public_key", "")
        )
        return

    # Fetch the pending action
    pending_action = await fetch_pending_action(action_id)

    if not pending_action:
        await send_cell_response(
            cell,
            transmitter.get("transmitter_id"),
            {"json": "Error: Action not found or already processed"},
            data.get("public_key", "")
        )
        return

    tool_name = pending_action.get("tool_name")
    try:
        parameters = json.loads(pending_action.get("parameter") or "{}")
        original_data = json.loads(pending_action.get("original_data") or "{}")
    except (json.JSONDecodeError, TypeError) as e:
        logging.error(f"[Approve] Malformed action data for {action_id}: {e}")
        await update_action_status(action_id, "failed", f"Malformed action data: {e}")
        await send_cell_response(
            cell,
            transmitter.get("transmitter_id"),
            {"json": f"Error: Action {action_id} has corrupted data and was marked as failed."},
            data.get("public_key", "")
        )
        return
    operator = original_data.get("operator", f"operator_{operator}")
    is_multi_step = pending_action.get("is_multi_step", False)

    try:
        registry = await tool_registry.get_registry()

        if is_multi_step:
            # Multi-step action execution
            steps_str = pending_action.get("steps", "[]")
            try:
                steps = json.loads(steps_str) if steps_str else []
            except json.JSONDecodeError:
                steps = []

            if not steps:
                await update_action_status(action_id, "failed", "No steps found in multi-step action")
                await send_cell_response(
                    cell,
                    transmitter.get("transmitter_id"),
                    {"json": "Error: No steps found in multi-step action"},
                    data.get("public_key", "")
                )
                return

            logging.info(f"[Multi-Step Execution]: {len(steps)} steps for action {action_id}")

            step_responses = []
            for step in steps:
                step_order = step.get("step_order", len(step_responses) + 1)
                step_subject = step.get("subject", f"Step {step_order}")
                step_tool_name = step.get("tool_name", "")
                step_parameters = step.get("parameters", {})

                logging.info(f"[Step {step_order}]: {step_tool_name} - {step_subject}")

                try:
                    mcp_result = await registry.call_tool(step_tool_name, step_parameters, operator)

                    step_response = await convert_tool_result_to_natural_language(
                        step_tool_name,
                        mcp_result,
                        step_subject
                    )

                    step_responses.append({
                        "step_order": step_order,
                        "step_subject": step_subject,
                        "status": "success",
                        "response": step_response
                    })

                    logging.info(f"[Step {step_order} Complete]: {step_response}")

                except Exception as step_error:
                    logging.error(f"[Step {step_order} Failed]: {step_error}")

                    step_responses.append({
                        "step_order": step_order,
                        "step_subject": step_subject,
                        "status": "error",
                        "response": f"Error: {str(step_error)}"
                    })

                    # Stop on first error
                    summary = await generate_multi_step_summary(
                        pending_action.get("subject", ""),
                        step_responses
                    )

                    await update_action_status(action_id, "failed", summary)
                    await store_message(operator, "assistant", summary)

                    html_content = ""
                    try:
                        template = env.get_template("index.html")
                        html_content = template.render()
                    except Exception:
                        pass

                    await send_cell_response(
                        cell,
                        transmitter.get("transmitter_id"),
                        {"json": summary, "html": html_content},
                        data.get("public_key", "")
                    )
                    return

            # All steps completed successfully
            summary = await generate_multi_step_summary(
                pending_action.get("subject", ""),
                step_responses
            )

            await update_action_status(action_id, "success", summary)
            await store_message(operator, "assistant", summary)

            # Serve default page for multi-step results
            html_content = ""
            try:
                template = env.get_template("index.html")
                html_content = template.render()
            except Exception:
                pass

            await send_cell_response(
                cell,
                transmitter.get("transmitter_id"),
                {"json": summary, "html": html_content},
                data.get("public_key", "")
            )

        else:
            # Single-step action execution
            mcp_result = await registry.call_tool(tool_name, parameters, operator)
            logging.info(f"[Approved Tool Result]: {mcp_result}")

            tool_response = await convert_tool_result_to_natural_language(
                tool_name,
                mcp_result,
                pending_action.get("subject", "")[:50]
            )

            await update_action_status(action_id, "success", tool_response)
            await store_message(operator, "assistant", tool_response)

            # Extract page to serve and template data from tool result
            html_content = ""
            page_to_serve = "index.html"
            template_data = {}
            try:
                content_items = mcp_result.get("content", [])
                if content_items:
                    result_text = content_items[0].get("text", "")
                    try:
                        result_data = json.loads(result_text)
                        page_to_serve = result_data.get("page", "index.html")
                        template_data = result_data
                    except (json.JSONDecodeError, AttributeError):
                        pass
                template = env.get_template(page_to_serve)
                html_content = template.render(**template_data)
                logging.info(f"[Serving Page]: {page_to_serve}")
            except Exception as e:
                logging.warning(f"Failed to load page {page_to_serve}: {e}")
                try:
                    template = env.get_template("index.html")
                    html_content = template.render()
                except Exception:
                    pass

            await send_cell_response(
                cell,
                transmitter.get("transmitter_id"),
                {"json": tool_response, "html": html_content},
                data.get("public_key", "")
            )

    except Exception as tool_error:
        logging.error(f"Approved tool execution failed: {tool_error}")
        error_message = "I apologize, but I encountered an issue while processing your request."

        await update_action_status(action_id, "failed", str(tool_error))

        await send_cell_response(
            cell,
            transmitter.get("transmitter_id"),
            {"json": error_message},
            data.get("public_key", "")
        )


async def handle_decline(cell, transmitter: dict):
    """Handle operator declining a pending action

    Payload: {type: "decline", action_id: X}
    """
    data = transmitter.get("data", {})
    action_id = data.get("action_id")
    operator = transmitter.get("operator", "anonymous_operator")

    logging.info(f"[Operator {operator}] Declining action {action_id}")

    if not action_id:
        await send_cell_response(
            cell,
            transmitter.get("transmitter_id"),
            {"json": "Error: action_id is required"},
            data.get("public_key", "")
        )
        return

    # Fetch the pending action
    pending_action = await fetch_pending_action(action_id)

    if not pending_action:
        await send_cell_response(
            cell,
            transmitter.get("transmitter_id"),
            {"json": "Error: Action not found or already processed"},
            data.get("public_key", "")
        )
        return

    # Update action status to dismissed
    await update_action_status(action_id, "dismissed", "Declined by operator")

    original_data = json.loads(pending_action.get("original_data", "{}"))
    operator = original_data.get("operator", f"operator_{operator}")

    decline_message = "No problem, I've cancelled that action. Is there anything else I can help you with?"

    await store_message(operator, "assistant", decline_message)

    await send_cell_response(
        cell,
        transmitter.get("transmitter_id"),
        {"json": decline_message},
        data.get("public_key", "")
    )


async def handle_prompt(cell, transmitter: dict):
    """Handle user prompt with tool routing

    This endpoint:
    - Routes user messages to the appropriate tool via LLM
    - Can use operator-accessible tools in a conversational flow
    - Returns both the JSON response and the matching HTML template
    """
    data = transmitter.get("data", {})
    prompt = data.get("prompt", "")
    operator = transmitter.get("operator", "anonymous_operator")

    logging.info(f"[User {operator}]: {prompt}")

    if not prompt:
        await send_cell_response(
            cell,
            transmitter.get("transmitter_id"),
            {"json": "Error: No prompt provided"},
            data.get("public_key", "")
        )
        return

    try:
        history = await fetch_latest_messages(operator, limit=10)

        # Get all available tools for operator use
        registry = await tool_registry.get_registry()
        operator_tools = await registry.get_all_tools()

        # Build tool descriptions for LLM
        tool_info_list = []
        for tool in operator_tools:
            input_schema = tool.get("inputSchema", {})
            properties = input_schema.get("properties", {})
            required_params = input_schema.get("required", [])

            params_desc = []
            for param_name, param_def in properties.items():
                if param_name == "operator":
                    continue
                is_required = "(REQUIRED)" if param_name in required_params else "(optional)"
                param_type = param_def.get('type', 'string')
                params_desc.append(f"  - {param_name} ({param_type}) {is_required}: {param_def.get('description', '')}")

            package_info = ""
            if tool.get("package_name"):
                package_info = f"\nPackage: {tool['package_name']}"
                if tool.get("package_description"):
                    package_info += f" - {tool['package_description']}"

            tool_info = f"""Tool: {tool['name']}
Description: {tool.get('description', 'No description')}{package_info}
Parameters:
{chr(10).join(params_desc) if params_desc else '  none'}"""
            tool_info_list.append(tool_info)

        tools_context = "\n\n".join(tool_info_list) if tool_info_list else "No tools available."

        # Fetch recent actions history for this operator (limit to last 10 to keep prompt concise)
        all_actions = await fetch_all_actions(operator=operator)
        recent_actions = all_actions[-10:] if len(all_actions) > 10 else all_actions
        actions_context = ""
        for action in recent_actions:
            actions_context += f"\n- [{action['status'].upper()}] {action['tool_name']}: {action['description']}"

        # Build system prompt — tool router (no indexed content, no answer action)
        system_prompt = textwrap.dedent(f"""
            You are a tool router. For every user request, you MUST select a tool from AVAILABLE TOOLS and fill in its parameters. You have NO knowledge of your own — all information must come from tool execution.

            AVAILABLE TOOLS:
            {tools_context}

            ACTIONS HISTORY (previous operator actions with their approval status):
            {actions_context if actions_context else "No previous actions."}

            RULES:
            - You MUST ALWAYS select a tool. There is no "answer" action — every request goes through a tool
            - If no tool matches the user's request → use action "clarify" and explain that this request cannot be handled
            - If a tool matches the request and ALL required parameters are present → use action "tool"
            - If a tool matches but required parameters are missing from the user's message → use action "clarify" and list exactly which parameters are needed
            - If the request needs exactly ONE tool call → use action "tool" with parameters filled in and NO "steps" key
            - ONLY if the request requires TWO OR MORE tool calls → use action "tool" with a "steps" array
            - Use ACTIONS HISTORY to understand what the operator has previously approved or declined

            RESPONSE FORMAT - You MUST respond with ONLY valid JSON, nothing else:

            For a single tool call:
            {{"action": "tool", "tool_name": "exact_tool_name", "parameters": {{"param1": "value1"}}, "message": "I'd like to [description]. Please confirm."}}

            For multiple tool calls (2+ steps):
            IMPORTANT: Put ALL actions inside the "steps" array. The top-level "parameters" MUST be empty {{}}.
            {{"action": "tool", "tool_name": "first_tool", "parameters": {{}}, "message": "I'd like to perform the following actions...", "steps": [{{"step_order": 1, "tool_name": "tool_a", "parameters": {{...}}, "subject": "First action"}}, {{"step_order": 2, "tool_name": "tool_b", "parameters": {{...}}, "subject": "Second action"}}]}}

            For asking clarification (tool identified but parameters missing):
            {{"action": "clarify", "message": "To proceed, I need: [list missing parameters]"}}

            TOOL RULES:
            - Copy the EXACT tool_name from AVAILABLE TOOLS
            - Include EVERY parameter marked (REQUIRED)
            - Parameter values MUST match the exact type described in the tool definition. If a parameter is described as a string (e.g. "comma-separated list"), pass it as a JSON string, NOT as a JSON array
            - The "message" field is a PROPOSAL shown to the operator BEFORE execution. It must describe what you WANT TO DO, not what was done. Use phrasing like "I'd like to..." or "Shall I...". NEVER write as if the action already happened
            - NEVER include "steps" when only ONE tool call is needed. "steps" is ONLY for 2+ tool calls
            - For multi-step actions (2+ calls), EVERY action must be in the "steps" array. Top-level "parameters" must be empty {{}}

            Respond with JSON only:
        """)

        messages = [{"role": "system", "content": system_prompt}]

        # Add conversation history
        for role, message in history:
            if role in ["user", "assistant"]:
                messages.append({"role": role, "content": message})

        # Add the user's question
        messages.append({"role": "user", "content": prompt})

        logging.info(f"[Agent]: Processing with {len(operator_tools)} tools available")

        # Generate response
        loop = asyncio.get_running_loop()

        def generate_response():
            response = create_chat_completion(
                messages=messages,
                max_tokens=MODEL_MAX_TOKENS,
                temperature=0.3
            )
            content = response['choices'][0]['message']['content']
            return content.strip()

        result_json = await loop.run_in_executor(None, generate_response)

        # Parse JSON response - find JSON anywhere in the response
        result_json = result_json.replace("```json", "").replace("```", "").strip()

        json_start = result_json.find('{')
        if json_start != -1:
            brace_count = 0
            end_index = json_start
            for i, char in enumerate(result_json[json_start:], start=json_start):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end_index = i + 1
                        break
            result_json = result_json[json_start:end_index]

        try:
            decision = json.loads(result_json)
        except json.JSONDecodeError:
            # Fallback: if JSON parsing fails, ask for clarification
            decision = {"action": "clarify", "message": "I'm sorry, I couldn't understand your request. Could you please rephrase it?"}

        action = decision.get("action", "tool")
        operator_message = decision.get("message", "I'm here to help. Could you please rephrase your question?")

        # Fallback: if the model put a tool name directly as the action value,
        # normalize it to action="tool" with the proper tool_name field
        tool_names_set = {t["name"] for t in operator_tools}
        if action not in ("clarify", "tool") and action in tool_names_set:
            logging.info(f"[Agent Decision Normalized]: action '{action}' is a tool name, normalizing to action='tool'")
            decision["tool_name"] = action
            decision.setdefault("parameters", {})
            action = "tool"
            decision["action"] = "tool"

        # Fallback: if model still chose "answer", treat as clarify
        if action == "answer":
            logging.info("[Agent Decision Normalized]: action 'answer' normalized to clarify")
            action = "clarify"
            decision["action"] = "clarify"
            if not decision.get("message"):
                operator_message = "I'm sorry, I couldn't find a matching tool for your request. Could you please rephrase it?"

        logging.info(f"[Agent Decision]: action={action}")

        # Load default HTML template (actual page comes from tool result in handle_approve)
        html_content = ""
        try:
            template = env.get_template("index.html")
            html_content = template.render()
        except Exception as e:
            logging.warning(f"Failed to load default template: {e}")

        if action == "clarify":
            logging.info(f"[Agent to User]: {operator_message}")

            await store_message(operator, "user", prompt)
            await store_message(operator, "assistant", operator_message)

            await send_cell_response(
                cell,
                transmitter.get("transmitter_id"),
                {"json": operator_message, "html": html_content},
                data.get("public_key", "")
            )

        elif action == "tool":
            tool_name = decision.get("tool_name", "")
            parameters = decision.get("parameters", {})
            steps = decision.get("steps", [])

            # Normalize: if model returned steps with only 1 entry, unwrap into single-step
            if len(steps) == 1:
                single = steps[0]
                tool_name = single.get("tool_name", tool_name)
                parameters = single.get("parameters", parameters)
                steps = []
                # Generate proper confirmation message from step details
                subject = single.get("subject", tool_name)
                param_details = ", ".join(f"{k}: {v}" for k, v in parameters.items()) if parameters else ""
                operator_message = f"I'd like to use {tool_name}: {subject}" + (f" ({param_details})" if param_details else "") + ". Please approve to proceed."
                logging.info(f"[Normalized]: unwrapped single-entry steps into single-step for {tool_name}")

            # Generate a proper confirmation message for single-step tool actions
            # if the model didn't provide one (avoids the generic "rephrase" fallback)
            if not steps and tool_name and operator_message == "I'm here to help. Could you please rephrase your question?":
                param_details = ", ".join(f"{k}: {v}" for k, v in parameters.items()) if parameters else ""
                operator_message = f"I'd like to use {tool_name}" + (f" ({param_details})" if param_details else "") + ". Please approve to proceed."

            # Normalize parameter names: map LLM mistakes to correct schema names
            def normalize_params(params, t_name):
                tool_schema = next((t.get("inputSchema", {}) for t in operator_tools if t["name"] == t_name), {})
                valid_names = set(tool_schema.get("properties", {}).keys())
                if not valid_names or all(k in valid_names for k in params):
                    return params
                normalized = {}
                for key, value in params.items():
                    if key in valid_names:
                        normalized[key] = value
                    else:
                        # Try to find matching schema param by substring
                        match = next((v for v in valid_names if v in key or key in v), None)
                        if match and match not in normalized:
                            logging.info(f"[Param Normalized]: '{key}' -> '{match}' for {t_name}")
                            normalized[match] = value
                        else:
                            normalized[key] = value
                return normalized

            parameters = normalize_params(parameters, tool_name)
            for step in steps:
                step["parameters"] = normalize_params(step.get("parameters", {}), step.get("tool_name", ""))

            is_multi_step = len(steps) >= 2

            logging.info(f"[Tool Action]: {tool_name} (multi_step={is_multi_step}, steps={len(steps)})")

            # Validate all tool names exist in operator tools
            tools_to_validate = [tool_name]
            if is_multi_step:
                tools_to_validate = [step.get("tool_name", "") for step in steps]

            invalid_tools = [t for t in tools_to_validate if t not in tool_names_set]

            if invalid_tools:
                logging.warning(f"Invalid tool(s) referenced: {invalid_tools}")
                operator_message = "No tool is currently installed to handle this action."

                await store_message(operator, "user", prompt)
                await store_message(operator, "assistant", operator_message)

                await send_cell_response(
                    cell,
                    transmitter.get("transmitter_id"),
                    {"json": operator_message, "html": html_content},
                    data.get("public_key", "")
                )
                return

            # Check if all tools in this request have auto_approve enabled
            tool_meta_map = {t["name"]: t for t in operator_tools}
            if is_multi_step:
                all_auto_approve = all(tool_meta_map.get(s.get("tool_name", ""), {}).get("auto_approve", False) for s in steps)
            else:
                all_auto_approve = tool_meta_map.get(tool_name, {}).get("auto_approve", False)

            if all_auto_approve and not is_multi_step:
                # Auto-approve: execute tool immediately without pending action
                logging.info(f"[Auto-Approve]: executing {tool_name} directly")

                try:
                    mcp_result = await registry.call_tool(tool_name, parameters, operator)
                    logging.info(f"[Auto-Approve Result]: {mcp_result}")

                    tool_response = await convert_tool_result_to_natural_language(
                        tool_name,
                        mcp_result,
                        prompt[:50]
                    )

                    # Store action as completed
                    await store_action_entry(
                        operator=operator,
                        subject=prompt[:100],
                        context=f"Auto-approved tool execution",
                        original_data=json.dumps({"prompt": prompt, "operator": operator}),
                        tool_name=tool_name,
                        parameter=json.dumps(parameters),
                        is_multi_step=False,
                        status="success",
                        response=tool_response
                    )

                    await store_message(operator, "user", prompt)
                    await store_message(operator, "assistant", tool_response)

                    # Extract page to serve and template data from tool result
                    page_to_serve = "index.html"
                    template_data = {}
                    try:
                        content_items = mcp_result.get("content", [])
                        if content_items:
                            result_text = content_items[0].get("text", "")
                            try:
                                result_data = json.loads(result_text)
                                page_to_serve = result_data.get("page", "index.html")
                                template_data = result_data
                            except (json.JSONDecodeError, AttributeError):
                                pass
                    except Exception:
                        pass

                    auto_html = ""
                    try:
                        template = env.get_template(page_to_serve)
                        auto_html = template.render(**template_data)
                        logging.info(f"[Serving Page]: {page_to_serve}")
                    except Exception as e:
                        logging.warning(f"Failed to load page {page_to_serve}: {e}")
                        try:
                            template = env.get_template("index.html")
                            auto_html = template.render()
                        except Exception:
                            pass

                    await send_cell_response(
                        cell,
                        transmitter.get("transmitter_id"),
                        {"json": tool_response, "html": auto_html},
                        data.get("public_key", "")
                    )

                except Exception as auto_error:
                    logging.error(f"Auto-approve tool execution failed: {auto_error}")
                    error_message = "I encountered an issue while processing your request."

                    await store_message(operator, "user", prompt)
                    await store_message(operator, "assistant", error_message)

                    await send_cell_response(
                        cell,
                        transmitter.get("transmitter_id"),
                        {"json": error_message, "html": html_content},
                        data.get("public_key", "")
                    )

            else:
                # Requires approval: create pending action
                action_id = await store_action_entry(
                    operator=operator,
                    subject=prompt[:100],
                    context=f"User {operator} tool suggestion",
                    original_data=json.dumps({"prompt": prompt, "operator": operator}),
                    tool_name=tool_name,
                    parameter=json.dumps(parameters),
                    is_multi_step=is_multi_step,
                    steps=json.dumps(steps) if is_multi_step else None,
                    status="pending"
                )

                logging.info(f"[Pending Action Created]: ID {action_id} for tool {tool_name} (multi_step={is_multi_step})")

                # For multi-step actions, build a confirmation message listing all steps
                if is_multi_step and steps:
                    step_descriptions = []
                    for i, step in enumerate(steps, 1):
                        subject = step.get("subject", step.get("tool_name", "Unknown"))
                        params = step.get("parameters", {})
                        param_details = ", ".join(f"{k}: {v}" for k, v in params.items()) if params else "no parameters"
                        step_descriptions.append(f"{i}. [{step.get('tool_name', 'unknown')}] {subject} ({param_details})")
                    step_word = "step" if len(steps) == 1 else "steps"
                    operator_message = f"I'd like to perform the following {len(steps)} {step_word}:\n" + "\n".join(step_descriptions) + "\n\nPlease approve to proceed."

                # Try to render a preview page for the pending action
                preview_html = html_content
                tool_info = tool_meta_map.get(tool_name, {})
                package_name = tool_info.get("package_name", "").lower()
                if package_name and not is_multi_step:
                    # Try common template naming patterns
                    for template_name in [f"{package_name}.html", f"{package_name}s.html", f"{tool_name}.html"]:
                        try:
                            preview_template = env.get_template(template_name)
                            preview_data = {**parameters, "preview": True, "message": operator_message}
                            preview_html = preview_template.render(**preview_data)
                            logging.info(f"[Preview Page]: {template_name}")
                            break
                        except Exception:
                            continue

                await store_message(operator, "user", prompt)
                await store_message(operator, "assistant", operator_message)

                await send_cell_response(
                    cell,
                    transmitter.get("transmitter_id"),
                    {"json": operator_message, "html": preview_html, "action_id": action_id},
                    data.get("public_key", "")
                )

        else:
            await store_message(operator, "user", prompt)
            await store_message(operator, "assistant", operator_message)

            await send_cell_response(
                cell,
                transmitter.get("transmitter_id"),
                {"json": operator_message, "html": html_content},
                data.get("public_key", "")
            )

    except Exception as e:
        logging.error(f"Error handling prompt: {e}")
        import traceback
        logging.error(traceback.format_exc())

        await send_cell_response(
            cell,
            transmitter.get("transmitter_id"),
            {"json": "I apologize, but I'm having trouble processing your request. Please try again."},
            data.get("public_key", "")
        )


async def handle_get_tools(cell, transmitter: dict):
    """Handle get tools request and return config data for each tool_id and all tasks"""
    data = transmitter.get("data", {})
    logging.info("Fetching all tool configs and tasks...")

    try:
        tools_dir = "./tools"
        tools_by_id = {}

        if os.path.exists(tools_dir):
            for filename in os.listdir(tools_dir):
                if filename.endswith(".config"):
                    tool_id = filename[:-7]
                    config_path = os.path.join(tools_dir, filename)

                    try:
                        with open(config_path, 'r') as f:
                            config_content = f.read()
                            config_json = json.loads(config_content)

                        tools_by_id[tool_id] = config_json

                    except json.JSONDecodeError as e:
                        logging.warning(f"Failed to parse config file {filename}: {e}")
                    except Exception as e:
                        logging.warning(f"Failed to read config file {filename}: {e}")

        logging.info(f"Retrieved {len(tools_by_id)} tool configs")

        response_data = {
            "tools": tools_by_id
        }

        await send_cell_response(
            cell,
            transmitter.get("transmitter_id"),
            {"json": response_data},
            data.get("public_key", "")
        )

    except Exception as e:
        logging.error(f"Error fetching tools and tasks: {e}")
        import traceback
        logging.error(traceback.format_exc())

        await send_cell_response(
            cell,
            transmitter.get("transmitter_id"),
            {"json": {"error": str(e)}},
            data.get("public_key", "")
        )

async def handle_install_tool(cell, transmitter: dict):
    """Handle adding new Tool tool from registry and restart agent"""
    data = transmitter.get("data", {})
    tool_id = data.get("tool_id", "")
    variables = data.get("variables", "")
    cell_id = transmitter.get("operator", "default_user")

    if not tool_id:
        logging.error("No tool_id provided")
        return

    try:
        tools = await cell.list_tools()
        logging.info(f"Available tools count: {len(tools)}")

        tool = None
        for t in tools:
            if t.get("tool_id") == tool_id:
                tool = t
                break

        if not tool:
            error_msg = f"Tool with ID '{tool_id}' not found"
            logging.error(error_msg)
            return

        script = tool.get("script", "")
        config = tool.get("config", "")
        author = tool.get("author", "Unknown")
        tool_name = tool.get("name", tool_id)

        if not script:
            error_msg = f"Tool '{tool_id}' has no script content"
            logging.error(error_msg)
            return

        tools_dir = "./tools"
        os.makedirs(tools_dir, exist_ok=True)

        final_script = script
        if variables:
            if isinstance(variables, str):
                try:
                    variables = json.loads(variables)
                except json.JSONDecodeError:
                    logging.warning(f"Could not parse variables as JSON: {variables}")
                    variables = {}

            if isinstance(variables, dict) and variables:
                variable_lines = []
                for var_name, var_value in variables.items():
                    escaped_value = str(var_value).replace('"', '\\"')
                    variable_lines.append(f'{var_name} = "{escaped_value}"')

                variables_block = '\n'.join(variable_lines) + '\n\n'

                script_lines = script.split('\n')
                insert_position = 0

                for i, line in enumerate(script_lines):
                    stripped = line.strip()
                    if stripped.startswith('#!') or stripped.startswith('# -*-') or stripped.startswith('#-*-'):
                        insert_position = i + 1
                    elif stripped and not stripped.startswith('#'):
                        break

                script_lines.insert(insert_position, variables_block.rstrip('\n'))
                final_script = '\n'.join(script_lines)
                logging.info(f"Injected {len(variables)} variable(s) into script: {list(variables.keys())}")

        script_filename = f"{tool_id}.py"
        script_path = os.path.join(tools_dir, script_filename)

        with open(script_path, 'w') as f:
            f.write(final_script)

        logging.info(f"Tool script saved to {script_path}")

        if config:
            config_filename = f"{tool_id}.config"
            config_path = os.path.join(tools_dir, config_filename)

            with open(config_path, 'w') as f:
                f.write(config)

            logging.info(f"Tool config saved to {config_path}")

        logging.info(f"Tool '{tool_id}' successfully added to tools")

        logging.info("Restarting agent to load new tool...")

        await cell.close()
        await asyncio.sleep(1)
        os.execv(sys.executable, [sys.executable] + sys.argv)

    except Exception as e:
        error_msg = f"Error adding tool: {str(e)}"
        logging.error(error_msg)
        import traceback
        logging.error(traceback.format_exc())


async def handle_delete_tool(cell, transmitter: dict):
    """Handle deleting tool from tools directory and restart agent"""
    data = transmitter.get("data", {})
    tool_id = data.get("tool_id", None)
    cell_id = transmitter.get("operator", "default_user")

    if not tool_id:
        logging.error("No tool_id provided for deletion")
        return

    try:
        tools_dir = "./tools"
        script_filename = f"{tool_id}.py"
        config_filename = f"{tool_id}.config"
        script_path = os.path.join(tools_dir, script_filename)
        config_path = os.path.join(tools_dir, config_filename)

        files_deleted = []

        if os.path.exists(script_path):
            os.remove(script_path)
            files_deleted.append(script_filename)
            logging.info(f"✅ Deleted tool script: {script_path}")

        if os.path.exists(config_path):
            os.remove(config_path)
            files_deleted.append(config_filename)
            logging.info(f"✅ Deleted tool config: {config_path}")

        if not files_deleted:
            logging.warning(f"Tool '{tool_id}' not found")
            return

        logging.info(f"Tool '{tool_id}' successfully deleted. Files removed: {', '.join(files_deleted)}")

        logging.info("Restarting agent...")

        await cell.close()

        await asyncio.sleep(1)

        os.execv(sys.executable, [sys.executable] + sys.argv)

    except Exception as e:
        logging.error(f"Error deleting tool: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())


# Message Routing

def is_authorized_internal_cell(operator: str, server_host: str) -> bool:
    """Check if operator is authorized to access internal handlers

    Authorization is based on the server's cell_id (host):
    - If server is 'neuronum.net::cell', authorized operators are:
      - 'neuronum.net::cell' (the business cell itself)
      - '*@neuronum.net::cell' (employees with cells from that domain)

    Args:
        operator: The cell_id of the request sender
        server_host: The cell_id of this server (cell.host)

    Returns:
        True if operator belongs to this organization, False otherwise
    """
    if not operator or not server_host:
        return False

    # Extract domain from server host (e.g., 'neuronum.net' from 'neuronum.net::cell')
    if not server_host.endswith("::cell"):
        return False

    server_domain = server_host[:-6]  # Remove '::cell'

    # Check if operator is the business cell itself
    if operator == server_host:
        return True

    # Check if operator is an employee (email@domain::cell)
    if operator.endswith("::cell"):
        operator_id = operator[:-6]  # Remove '::cell'

        # Employee pattern: user@domain
        if "@" in operator_id:
            email_domain = operator_id.split("@")[-1]
            if email_domain == server_domain:
                return True

    return False


# Handlers that community/operators are allowed to access
CUSTOMER_ALLOWED_HANDLERS = {
    "prompt",
    "approve",
    "decline",
    "get_agent_status",
    "get_index"
}


async def route_message(cell, transmitter: dict):
    """Route incoming messages to appropriate handlers with access control"""
    try:
        data = transmitter.get("data", {})
        message_type = data.get("type", None)
        operator = transmitter.get("operator", "")

        # Get the server's cell_id to determine authorized cells
        server_host = cell.host or cell.env.get("HOST", "")

        # Check if operator is authorized for internal access
        is_internal = is_authorized_internal_cell(operator, server_host)

        # If not internal and trying to access internal handlers, deny access
        if not is_internal and message_type not in CUSTOMER_ALLOWED_HANDLERS:
            logging.warning(f"Access denied: '{operator}' attempted to access '{message_type}' (not authorized for {server_host})")
            await send_cell_response(
                cell,
                transmitter.get("transmitter_id"),
                {"json": "Access denied: This endpoint is not available."},
                data.get("public_key", "")
            )
            return

        handlers = {
            "get_index": lambda: handle_get_index(cell, transmitter),
            "get_agent_status": lambda: handle_get_status(cell, transmitter),
            "get_actions": lambda: handle_get_actions(cell, transmitter),
            "prompt": lambda: handle_prompt(cell, transmitter),
            "approve": lambda: handle_approve(cell, transmitter),
            "decline": lambda: handle_decline(cell, transmitter),
            "get_tools": lambda: handle_get_tools(cell, transmitter),
            "install_tool": lambda: handle_install_tool(cell, transmitter),
            "delete_tool": lambda: handle_delete_tool(cell, transmitter)
        }

        handler = handlers.get(message_type)
        if handler:
            await handler()
        else:
            logging.warning(f"Unknown message type: {message_type}")
    except Exception as e:
        logging.error(f"Error routing message: {e}")
        import traceback
        logging.error(traceback.format_exc())

def _task_done_callback(task: asyncio.Task):
    """Log unhandled exceptions from fire-and-forget tasks"""
    if task.cancelled():
        return
    exc = task.exception()
    if exc:
        logging.error(f"Unhandled exception in message task: {exc}", exc_info=exc)

async def process_cell_messages(cell):
    """Main message processing loop for cell — dispatches messages concurrently"""
    async for transmitter in cell.sync():
        task = asyncio.create_task(route_message(cell, transmitter))
        task.add_done_callback(_task_done_callback)

# Main Function

async def server_main():
    """Main server logic"""
    cell = None
    try:
        logging.info("Setting up infrastructure...")
        await setup_infrastructure()

        logging.info("Initializing database...")
        await initialize_database()

        await install_tool_requirements()

        logging.info("Connecting to Neuronum network...")
        cell = await setup_cell_connection()
        logging.info(f"Connected as Cell: {cell.env.get('HOST') or cell.host}")

        logging.info("Loading Tools...")
        registry = await tool_registry.initialize_registry(cell, logging)

        available_tools = await registry.get_all_tools()
        tool_count = len(available_tools)
        server_info = await registry.get_server_info()
        logging.info(f"Tool Registry initialized with {server_info['total_servers']} servers and {tool_count} tools")

        logging.info(f"Agent started as Cell: {cell.host}")
        logging.info(f"Agent running with {tool_count} Tools")

        if not cell.host.startswith("neuronumagent"):
            await cell.stream(cell.host, {"json": "ping"})

        await process_cell_messages(cell)
    finally:
        if cell is not None:
            try:
                await cell.close()
                logging.info("Cell connection closed successfully")
            except Exception as e:
                logging.error(f"Error closing cell connection: {e}")

async def main():
    """Main entry point"""
    await server_main()


if __name__ == "__main__":
    asyncio.run(main())
