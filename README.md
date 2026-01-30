### **Neuronum Server**
Neuronum Server is an agent-wrapper that transforms your model into an agentic backend server that can interact with the [Neuronum Client API](#neuronum-client-api) and installed tools

> **Protocol Note:** The Neuronum SDK is powered by an end-to-end encrypted communication protocol based on public/private key pairs derived from a randomly generated 12-word mnemonic. All data is relayed through neuronum.net, providing secure communication without the need to set up public web servers or expose your infrastructure to the public internet.

> ⚠️ **Development Status:** The Neuronum SDK is currently in early stages of development and is **not production-ready**. It is intended for development, testing, and experimental purposes only. Do not use in production environments or for critical applications.
------------------

### **Requirements**
- Python >= 3.8
- **Linux/NVIDIA GPU:** CUDA-compatible GPU + CUDA Toolkit (uses vLLM for model serving)
- **macOS Apple Silicon:** Ollama installed (uses Ollama for model serving)

------------------

### **Connect To Neuronum**
**Installation**

Create and activate a virtual environment:
```sh
python3 -m venv ~/neuronum-venv
source ~/neuronum-venv/bin/activate
```

Install the Neuronum SDK:
```sh
pip install neuronum==2026.01.0.dev1
```

> **Note:** Always activate this virtual environment (`source ~/neuronum-venv/bin/activate`) before running any `neuronum` commands.

**Create a Neuronum Cell**
<br>The Neuronum Cell is your secure identity to interact with the Network
```sh
neuronum create-cell
```

**Connect your Cell**
```sh
neuronum connect-cell
```

------------------


**Start the Server**

```sh
neuronum start-server
```

This command will:
- Clone the neuronum-server repository (if not already present)
- Detect your hardware platform (Apple Silicon or NVIDIA GPU)
- Create a Python virtual environment
- Install platform-specific dependencies
- **Apple Silicon:** Verify Ollama is installed, start the Ollama server, and pull the configured model
- **NVIDIA GPU:** Start the vLLM server in the background and wait for model loading
- Launch the Neuronum Server

**Viewing Logs**

```sh
tail -f neuronum-server/server.log
tail -f neuronum-server/vllm_server.log
```

**Check Server Status**

```sh
neuronum status
```

**Stopping the Server**

```sh
neuronum stop-server
```

**What the Server Does**

Once running, the server will:
- Connect to the Neuronum network using your Cell credentials
- Initialize a local SQLite database for conversation memory and auto-indexes files in the `templates/` directory
- Auto-discover and launch any MCP servers in the `tools/` directory
- Process messages from clients via the Neuronum network

**Server Configuration**

The server can be customized by editing the `neuronum-server/server.config` file. Here are the available options:

**File Paths:**
```python
LOG_FILE = "server.log"              # Server log file location
DB_PATH = "agent_memory.db"          # SQLite database for conversations and knowledge
TEMPLATES_DIR = "./templates"        # HTML templates to auto-index on startup and serve
```

**Model Configuration:**
```python
MODEL_MAX_TOKENS = 512               # Maximum tokens in responses (higher = longer answers)
MODEL_TEMPERATURE = 0.3              # Creativity (0.0 = deterministic, 1.0 = creative)
MODEL_TOP_P = 0.85                   # Nucleus sampling (lower = more predictable)
```

**vLLM Server (NVIDIA GPU):**
```python
VLLM_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"  # Model to load
                                               # Examples: "Qwen/Qwen2.5-1.5B-Instruct",
                                               #           "meta-llama/Llama-3.2-3B-Instruct"
VLLM_HOST = "127.0.0.1"              # Server host (127.0.0.1 = local only)
VLLM_PORT = 8000                     # Server port
VLLM_API_BASE = "http://127.0.0.1:8000/v1"  # Full API URL
```

**Ollama (Apple Silicon):**
```python
OLLAMA_MODEL_NAME = "llama3.1:8b"    # Model to load
                                     # Examples: "llama3.2:3b", "qwen2.5:3b", "qwen2.5:7b"
OLLAMA_API_BASE = "http://127.0.0.1:11434/v1"  # Ollama API URL (default port: 11434)
```

**Conversation & Knowledge:**
```python
CONVERSATION_HISTORY_LIMIT = 10      # Recent messages to include in context
KNOWLEDGE_RETRIEVAL_LIMIT = 5        # Max knowledge chunks to retrieve
FTS5_STOPWORDS = {...}               # Words to exclude from knowledge search
```

After modifying the configuration, restart the server for changes to take effect:
```sh
neuronum stop-server
neuronum start-server
```

