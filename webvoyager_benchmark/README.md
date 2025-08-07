# Browser Automation Agent

A simple, clean browser automation agent inspired by nanoGPT's simplicity and the effectiveness of Browser-Use, Notte, and Magnitude implementations.

## Features

- **DOM-first approach** with element numbering for easy LLM reference
- **Patchright browser** with anti-detection features
- **Ollama integration** for local LLM inference (qwen3:32b or gpt-oss:20b)
- **Type-safe** Python with full type annotations
- **Simple architecture** inspired by nanoGPT

## Installation

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) package manager
- [ollama](https://ollama.ai) for LLM inference

### Setup

1. Create and activate virtual environment:
```bash
uv venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install dependencies:
```bash
uv pip install -r requirements.txt
```

3. Install browser:
```bash
patchright install chromium
```

4. Install and run ollama model:
```bash
# Install ollama (if not already installed)
# See: https://ollama.ai/download

# Pull the model (choose one)
ollama pull qwen3:32b
# or
ollama pull gpt-oss:20b

# Start ollama server
ollama serve
```

5. Test installation:
```bash
python test_installation.py
```

## Usage

### Basic Example

```python
from main import BrowserAgent
import asyncio

async def run():
    agent = BrowserAgent(task="Search for Python documentation")
    result = await agent.run(start_url="https://www.google.com")
    print(f"Success: {result['success']}")
    print(f"Steps taken: {result['steps']}")

asyncio.run(run())
```

### Run the Demo

```bash
python main.py
```

This will run a simple demo task searching for "WebVoyager benchmark" on Google.

### Configuration

Edit the configuration section in `main.py`:

```python
# Model configuration
MODEL_NAME = "qwen3:32b"  # or "gpt-oss:20b"
OLLAMA_HOST = "http://localhost:11434"

# Browser configuration
HEADLESS = False  # Set to True for production
VIEWPORT_WIDTH = 1280
VIEWPORT_HEIGHT = 800

# Agent configuration
MAX_STEPS = 30
TIMEOUT_MS = 30000
```

## Type Checking

The project uses `ty` for type checking:

```bash
# Check all Python files
ty check

# Check specific file
ty check main.py
```

## Architecture

### Core Components

1. **BrowserController** - Handles browser automation using patchright
   - DOM extraction via JavaScript injection
   - Element numbering and XPath generation
   - Action execution (click, type, scroll, navigate)

2. **LLMReasoner** - Manages LLM interactions
   - Structured JSON prompts
   - Action decision based on current state
   - History tracking for context

3. **BrowserAgent** - Main orchestration
   - Task execution loop
   - State management
   - Result tracking

### DOM Extraction Strategy

The agent uses JavaScript injection to:
- Find all interactive elements (buttons, links, inputs)
- Number each element for easy reference
- Generate XPath selectors for reliable targeting
- Filter invisible elements

### Action Types

- `navigate` - Go to a URL
- `click` - Click an element by index
- `type` - Type text into an input
- `scroll` - Scroll the page
- `extract` - Extract text from the page
- `wait` - Wait for page load
- `done` - Mark task complete

## Development

### Project Structure

```
jan/
├── main.py              # Main agent implementation
├── strategies.md        # Analysis of SOTA implementations
├── requirements.txt     # Python dependencies
├── pyproject.toml      # Project configuration
├── test_installation.py # Installation test script
├── README.md           # This file
└── .venv/              # Virtual environment
```

### Future Enhancements

See `strategies.md` for detailed analysis and future improvements including:
- Perception layer for semantic DOM descriptions
- UI-TARS integration for visual grounding
- Multi-step planning
- Site-specific handlers
- Advanced error recovery

## Troubleshooting

### Ollama Connection Error
```bash
# Make sure ollama is running
ollama serve

# Check available models
ollama list
```

### Browser Launch Error
```bash
# Reinstall chromium
patchright install chromium --force
```

### Type Checking Errors
```bash
# Make sure virtual environment is activated
source .venv/bin/activate
ty check main.py
```

## License

MIT

## Acknowledgments

- Inspired by [nanoGPT](https://github.com/karpathy/nanoGPT)'s simplicity
- Based on patterns from [Browser-Use](https://github.com/browser-use/browser-use), [Notte](https://github.com/nottelabs/notte), and [Magnitude](https://github.com/magnitudedev/magnitude)