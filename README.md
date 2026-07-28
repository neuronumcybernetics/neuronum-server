<h1 align="center">
  <img src="https://neuronum.net/static/logo_new.png" alt="Neuronum" width="80">
</h1>
<h4 align="center">Neuronum Server</h4>

<p align="center">
  <a href="https://neuronum.net">
    <img src="https://img.shields.io/badge/Website-Neuronum-blue" alt="Website">
  </a>
  <a href="https://neuronum.net/docs">
    <img src="https://img.shields.io/badge/Docs-Read%20now-green" alt="Documentation">
  </a>
  <a href="https://pypi.org/project/neuronum/">
    <img src="https://img.shields.io/pypi/v/neuronum.svg" alt="PyPI Version">
  </a><br>
  <img src="https://img.shields.io/badge/Python-3.8%2B-yellow" alt="Python Version">
  <a href="https://github.com/neuronumcybernetics/cell-sdk-python/blob/main/LICENSE.md">
    <img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License">
  </a>
</p>

------------------

### **About**

Neuronum Server is a lightweight AI Agent runtime for communicating across the Neuronum network. Plug your Agent into it and start automating your tasks through conversational Agent-to-Agent and Agent-to-Client connections

> ⚠️ **Development Status:** The Neuronum SDK is currently in beta and is **not production-ready**. It is intended for development, testing, and experimental purposes only. Do not use in production environments or for critical applications.

------------------

### **Requirements**
- Python >= 3.8

------------------

### **Running the Server**

Follow these steps to get the neuronum-server running:

1. **Clone the repository:**
```sh
git clone https://github.com/neuronumcybernetics/agent-server
cd agent-server
```

2. **Install the Neuronum SDK:**
```sh
pip install neuronum
```

3. **Set up your Cell (your digital identity on the Neuronum network):**
   - If you don't have a Cell yet:
   ```sh
   neuronum create-cell
   ```
   - If you already have a Cell and want to connect it to this device:
   ```sh
   neuronum connect-cell
   ```

4. **Install the server dependencies:**
```sh
pip install -r requirements.txt
```

5. **Configure your OpenAI-compatible API key:**
   ```
   - Edit `.env` and set your `CLIENT_API_KEY` (and optionally `MODEL_BASE_URL` and `MODEL_NAME`):
   ```
   Supported providers include OpenAI, Groq, OpenRouter, or any OpenAI-compatible endpoint.

6. **Start the server:**
```sh
python server.py
```

-----------------

### **Full Documentation**
Visit the [Neuronum Docs](https://neuronum.net/docs) for the complete SDK reference.
