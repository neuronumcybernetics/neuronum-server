<h1 align="center">
  <img src="https://neuronum.net/static/logo_new.png" alt="Neuronum" width="80">
</h1>
<h4 align="center">Neuronum SDK</h4>

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

Neuronum is built around the Secure Agent Session (SAS). An end-to-end encrypted channel designed for agent-to-client and agent-to-agent communication across businesses, partners, and customers. A session connects two parties to automate data exchange, take actions, and coordinate tasks without manual integration, custom APIs, or file transfers.

The SDK handles encryption, identity, and delivery. You write the agent logic.

> ⚠️ **Development Status:** The Neuronum SDK is currently in beta and is **not production-ready**. It is intended for development, testing, and experimental purposes only. Do not use in production environments or for critical applications.

------------------

### **Requirements**
- Python >= 3.8

------------------

### **Running the Server**

Follow these steps to get the neuronum-server running:

1. **Clone the repository:**
```sh
git clone https://github.com/neuronumcybernetics/neuronum-server
cd neuronum-server
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
   CLIENT_API_KEY=your_api_key_here
   MODEL_BASE_URL=https://api.openhippo.io/v1
   MODEL_NAME=openai/gpt-oss-120b
   ```
   Supported providers include OpenAI, Groq, OpenRouter, or any OpenAI-compatible endpoint.

6. **Start the server:**
```sh
python server.py
```

-----------------

### **Full Documentation**
Visit the [Neuronum Docs](https://neuronum.net/docs) for the complete SDK reference.
# neuronum-server
