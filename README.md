### **Neuronum Server**
Neuronum Server is an agent-wrapper that transforms your model into an agentic backend server that can interact with the [Neuronum Client API](#neuronum-client-api) and installed tools

------------------

### **Requirements**
- Python >= 3.8
- CUDA-compatible GPU (for Neuronum Server)
- CUDA Toolkit (for Neuronum Server)

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
pip install neuronum==2025.12.0.dev6
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


**Setup the Server**

Clone the neuronum-server repository:
```sh
git clone https://github.com/neuronumcybernetics/neuronum-server.git
cd neuronum-server
```

Run the setup script:
```sh
bash start_neuronum_server.sh
```

The setup script will:
- Create a Python virtual environment
- Install all dependencies (vLLM, PyTorch, etc.)
- Start the vLLM server in the background
- Launch the Neuronum Server

**Viewing Logs**

```sh
tail -f server.log
tail -f vllm_server.log
```

**Stopping the Server**

```sh
bash stop_neuronum_server.sh
```

**What the Server Does**

Once running, the server will:
- Connect to the Neuronum network using your Cell credentials
- Initialize a local SQLite database for conversation memory and knowledge storage
- Auto-discover and launch any MCP servers in the `tools/` directory
- Process messages from clients via the Neuronum network
- Execute scheduled tasks defined in the `tasks/` directory

------------------
