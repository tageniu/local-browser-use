#!/usr/bin/env python3
"""
Simple Browser Automation Agent
Inspired by nanoGPT's simplicity
"""

import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path

# Third-party imports
from patchright.async_api import async_playwright, Page, Browser, BrowserContext
from pydantic import BaseModel, Field
import ollama
from groq import Groq
from loguru import logger
from dotenv import load_dotenv
from langfuse.openai import openai as langfuse_openai

# Load environment variables
load_dotenv()

# Langfuse will be auto-initialized when using langfuse_openai
langfuse_enabled = False
try:
    # Test if Langfuse credentials are available
    test_client = langfuse_openai.OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=os.getenv("GROQ_API_KEY", "test")
    )
    langfuse_enabled = True
    logger.info("Langfuse tracing enabled for LLM calls")
except Exception as e:
    logger.warning(f"Langfuse initialization failed, continuing without tracing: {e}")

# ============================================================================
# Configuration Section (nanoGPT style)
# ============================================================================

# LLM Provider configuration
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()  # "ollama" or "groq"
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-oss:120b")  # qwen3:32b, gpt-oss:20b, or gpt-oss:120b

# Provider-specific configuration
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# Model mapping for Groq (maps Ollama model names to Groq model names)
GROQ_MODEL_MAP = {
    "qwen3:32b": "qwen/qwen3-32b",
    "gpt-oss:20b": "openai/gpt-oss-20b",
    "gpt-oss:120b": "openai/gpt-oss-120b"
}

# Browser configuration
HEADLESS = False  # Set to True for production
VIEWPORT_WIDTH = 1280
VIEWPORT_HEIGHT = 800
USER_AGENT = None  # Use default

# DOM extraction configuration
MAX_ELEMENTS = 150  # Maximum interactive elements to extract
INCLUDE_ATTRIBUTES = ["id", "class", "href", "title", "alt", "placeholder", "value", "type"]

# DOM observation configuration (for debugging what the AI sees)
# Set SAVE_DOM_TO_FILE=true in .env to save DOM snapshots to files
# Set MAX_DOM_ELEMENTS_TO_SHOW=100 in .env to show more elements (default: 50)
# The LLM now receives:
#   1. Full DOM structure (hierarchical view of the page)
#   2. Actionable elements map with Chain-of-Thought reasoning about what each element does

# Agent configuration
MAX_STEPS = 30
TIMEOUT_MS = 30000  # 30 seconds
RETRY_ATTEMPTS = 3
THINK_TIME = 0.5  # Seconds to wait between actions

# Logging configuration
LOG_LEVEL = "INFO"
logger.remove()
logger.add(sys.stderr, level=LOG_LEVEL)

# ============================================================================
# Data Models
# ============================================================================

class ActionType(str, Enum):
    NAVIGATE = "navigate"
    CLICK = "click"
    TYPE = "type"
    SCROLL = "scroll"
    WAIT = "wait"
    EXTRACT = "extract"
    DONE = "done"

@dataclass
class DOMElement:
    """Represents an interactive DOM element"""
    index: int
    tag: str
    text: str
    attributes: Dict[str, str]
    xpath: str
    is_visible: bool = True
    is_clickable: bool = False
    is_input: bool = False
    parent_text: str = ""  # Context from parent element
    aria_label: str = ""   # Accessibility label

    def to_string(self) -> str:
        """Convert to LLM-friendly string representation"""
        text_preview = self.text[:50] + "..." if len(self.text) > 50 else self.text
        element_type = "input" if self.is_input else "button" if self.is_clickable else self.tag
        
        # Include relevant attributes
        attrs = []
        if self.attributes.get('id'):
            attrs.append(f"id='{self.attributes['id']}'")
        if self.attributes.get('class'):
            attrs.append(f"class='{self.attributes['class'][:30]}...'")
        if self.attributes.get('href'):
            attrs.append(f"href='{self.attributes['href'][:50]}...'")
        if self.attributes.get('placeholder'):
            attrs.append(f"placeholder='{self.attributes['placeholder']}'")
        
        attrs_str = f" ({', '.join(attrs)})" if attrs else ""
        
        return f"[{self.index}] {element_type}: {text_preview}{attrs_str}"
    
    def get_action_reasoning(self) -> str:
        """Generate reasoning about what this element does"""
        reasoning = []
        
        # Determine element purpose
        if self.is_input:
            input_type = self.attributes.get('type', 'text')
            placeholder = self.attributes.get('placeholder', '')
            name = self.attributes.get('name', '')
            
            if input_type == 'search' or 'search' in placeholder.lower() or 'search' in name.lower():
                reasoning.append("This is a search input field")
            elif input_type == 'email' or 'email' in placeholder.lower():
                reasoning.append("This is an email input field")
            elif input_type == 'password':
                reasoning.append("This is a password input field")
            else:
                reasoning.append(f"This is a {input_type} input field")
            
            if placeholder:
                reasoning.append(f"expecting: '{placeholder}'")
        
        elif self.tag == 'a' and self.attributes.get('href'):
            href = self.attributes.get('href', '')
            if href.startswith('http'):
                domain = href.split('/')[2] if len(href.split('/')) > 2 else 'external site'
                reasoning.append(f"This link navigates to {domain}")
            elif href.startswith('#'):
                reasoning.append("This link navigates to a section on the current page")
            elif href == '#' or href == 'javascript:void(0)':
                reasoning.append("This link triggers a JavaScript action")
            else:
                reasoning.append(f"This link navigates to {href}")
        
        elif self.is_clickable:
            # Analyze button text and context
            text_lower = self.text.lower()
            if 'submit' in text_lower or 'send' in text_lower:
                reasoning.append("This button submits a form")
            elif 'search' in text_lower:
                reasoning.append("This button triggers a search")
            elif 'sign in' in text_lower or 'log in' in text_lower:
                reasoning.append("This button initiates login")
            elif 'sign up' in text_lower or 'register' in text_lower:
                reasoning.append("This button starts registration")
            elif 'next' in text_lower:
                reasoning.append("This button proceeds to the next step")
            elif 'back' in text_lower or 'previous' in text_lower:
                reasoning.append("This button goes back")
            else:
                reasoning.append(f"This button performs action: '{self.text}'")
        
        # Add aria-label insight if available
        if self.aria_label:
            reasoning.append(f"Accessibility: '{self.aria_label}'")
        
        return " | ".join(reasoning) if reasoning else "Purpose unclear from context"

@dataclass
class BrowserState:
    """Current state of the browser"""
    url: str
    title: str
    elements: List[DOMElement]
    full_dom: str = ""  # Full DOM structure
    screenshot: Optional[bytes] = None
    
    def to_prompt_string(self) -> str:
        """Convert state to prompt-friendly format"""
        # Get max elements to show from environment or use default
        max_show = int(os.getenv("MAX_DOM_ELEMENTS_TO_SHOW", "100"))
        
        # Create actionable elements map with reasoning
        action_map = []
        for element in self.elements[:max_show]:
            reasoning = element.get_action_reasoning()
            element_str = element.to_string()
            action_map.append(f"{element_str}\n   → Action: {reasoning}")
        
        action_map_str = "\n".join(action_map)
        
        # Add summary if there are more elements
        summary = ""
        if len(self.elements) > max_show:
            summary = f"\n... and {len(self.elements) - max_show} more elements (showing first {max_show} of {len(self.elements)} total)"
        
        # Count element types
        buttons = sum(1 for e in self.elements if e.is_clickable)
        inputs = sum(1 for e in self.elements if e.is_input)
        links = sum(1 for e in self.elements if e.tag == 'a')
        
        # Truncate full DOM if too long
        dom_preview = self.full_dom
        if len(dom_preview) > 3000:
            dom_preview = dom_preview[:3000] + "\n... (DOM truncated for brevity)"
        
        return f"""Current Page:
URL: {self.url}
Title: {self.title}

=== PAGE STRUCTURE (Full DOM) ===
{dom_preview}

=== ACTIONABLE ELEMENTS MAP ===
Element Summary:
- Total interactive elements: {len(self.elements)}
- Buttons/Clickable: {buttons}
- Input fields: {inputs}
- Links: {links}

Action Map (What each element does):
{action_map_str}{summary}
"""

class AgentAction(BaseModel):
    """Action to be performed by the agent"""
    reasoning: str = Field(description="Step-by-step reasoning for this action")
    action: ActionType = Field(description="Type of action to perform")
    element_index: Optional[int] = Field(None, description="Index of element to interact with")
    value: Optional[str] = Field(None, description="Value for type actions or extraction result")

# ============================================================================
# Browser Controller
# ============================================================================

class BrowserController:
    """Handles browser automation using patchright"""
    
    def __init__(self):
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None
        self.element_map: Dict[int, DOMElement] = {}
    
    async def start(self):
        """Launch browser and create page"""
        logger.info("Starting browser...")
        self.playwright = await async_playwright().start()
        
        # Use patchright's anti-detection features
        self.browser = await self.playwright.chromium.launch(
            headless=HEADLESS,
            args=["--disable-blink-features=AutomationControlled"]
        )
        
        self.context = await self.browser.new_context(
            viewport={"width": VIEWPORT_WIDTH, "height": VIEWPORT_HEIGHT},
            user_agent=USER_AGENT
        )
        
        self.page = await self.context.new_page()
        logger.info("Browser started successfully")
    
    async def close(self):
        """Close browser and cleanup"""
        if self.page:
            await self.page.close()
        if self.context:
            await self.context.close()
        if self.browser:
            await self.browser.close()
        if hasattr(self, 'playwright'):
            await self.playwright.stop()
        logger.info("Browser closed")
    
    async def navigate(self, url: str) -> bool:
        """Navigate to URL"""
        try:
            await self.page.goto(url, wait_until="domcontentloaded", timeout=TIMEOUT_MS)
            await self.page.wait_for_load_state("networkidle", timeout=5000)
            return True
        except Exception as e:
            logger.error(f"Navigation failed: {e}")
            return False
    
    async def extract_dom(self) -> Tuple[List[DOMElement], str]:
        """Extract both full DOM structure and interactive elements"""
        
        # Load JavaScript code from file
        js_file = Path(__file__).parent / "prompts" / "dom_extraction.js"
        js_template = js_file.read_text()
        # Replace MAX_ELEMENTS placeholder in the JavaScript
        js_code = js_template.replace('MAX_ELEMENTS', str(MAX_ELEMENTS))
        
        try:
            result = await self.page.evaluate(js_code)
            elements_data = result['elements']
            full_dom = result['fullDOM']
            
            elements = []
            self.element_map = {}
            
            for elem_data in elements_data:
                element = DOMElement(
                    index=elem_data['index'],
                    tag=elem_data['tag'],
                    text=elem_data['text'],
                    attributes=elem_data['attributes'],
                    xpath=elem_data['xpath'],
                    is_visible=elem_data['is_visible'],
                    is_clickable=elem_data['is_clickable'],
                    is_input=elem_data['is_input'],
                    parent_text=elem_data.get('parent_text', ''),
                    aria_label=elem_data.get('aria_label', '')
                )
                elements.append(element)
                self.element_map[element.index] = element
            
            logger.debug(f"Extracted {len(elements)} DOM elements and {len(full_dom)} chars of DOM structure")
            return elements, full_dom
            
        except Exception as e:
            logger.error(f"DOM extraction failed: {e}")
            return [], ""
    
    async def get_state(self) -> BrowserState:
        """Get current browser state"""
        url = self.page.url
        title = await self.page.title()
        elements, full_dom = await self.extract_dom()
        
        # Optionally capture screenshot
        screenshot = None
        # screenshot = await self.page.screenshot()
        
        return BrowserState(url=url, title=title, elements=elements, full_dom=full_dom, screenshot=screenshot)
    
    async def click_element(self, index: int) -> bool:
        """Click element by index"""
        if index not in self.element_map:
            logger.error(f"Element {index} not found")
            return False
        
        element = self.element_map[index]
        try:
            # Try to click using XPath
            await self.page.click(f'xpath={element.xpath}', timeout=5000)
            await asyncio.sleep(THINK_TIME)
            return True
        except Exception as e:
            logger.error(f"Click failed: {e}")
            # Fallback: try JavaScript click
            try:
                await self.page.evaluate(f"""
                    document.evaluate('{element.xpath}', document, null, 
                        XPathResult.FIRST_ORDERED_NODE_TYPE, null).singleNodeValue.click()
                """)
                return True
            except Exception as e:
                logger.error(f"Fallback click failed: {e}")
                return False
    
    async def type_text(self, index: int, text: str) -> bool:
        """Type text into element"""
        if index not in self.element_map:
            logger.error(f"Element {index} not found")
            return False
        
        element = self.element_map[index]
        try:
            # Clear existing text and type new
            await self.page.fill(f'xpath={element.xpath}', text)
            await asyncio.sleep(THINK_TIME)
            return True
        except Exception as e:
            logger.error(f"Type failed: {e}")
            return False
    
    async def scroll(self, direction: str = "down", amount: int = 500) -> bool:
        """Scroll the page"""
        try:
            if direction == "down":
                await self.page.evaluate(f"window.scrollBy(0, {amount})")
            else:
                await self.page.evaluate(f"window.scrollBy(0, -{amount})")
            await asyncio.sleep(THINK_TIME)
            return True
        except Exception as e:
            logger.error(f"Scroll failed: {e}")
            return False
    
    async def extract_text(self) -> str:
        """Extract all visible text from page"""
        try:
            text = await self.page.evaluate("""
                () => document.body.innerText || document.body.textContent || ''
            """)
            return text
        except Exception as e:
            logger.error(f"Text extraction failed: {e}")
            return ""

# ============================================================================
# LLM Reasoner
# ============================================================================

class LLMReasoner:
    """Handles LLM interactions using ollama or groq with Langfuse tracing"""
    
    def __init__(self, model: str = MODEL_NAME, provider: str = LLM_PROVIDER):
        self.provider = provider
        
        if provider == "groq":
            if not GROQ_API_KEY:
                raise ValueError("GROQ_API_KEY not found in environment variables")
            
            # Use Langfuse OpenAI wrapper for Groq (compatible with OpenAI format)
            if langfuse_enabled:
                self.client = langfuse_openai.OpenAI(
                    base_url="https://api.groq.com/openai/v1",
                    api_key=GROQ_API_KEY
                )
                self.using_langfuse = True
            else:
                self.client = Groq(api_key=GROQ_API_KEY)
                self.using_langfuse = False
            
            # Map model name to Groq model format
            self.model = GROQ_MODEL_MAP.get(model, model)
            logger.info(f"Using Groq with model: {self.model} (Langfuse: {self.using_langfuse})")
        else:
            self.client = ollama.Client(host=OLLAMA_HOST)
            self.model = model  # Ollama uses model names directly
            self.using_langfuse = False
            logger.info(f"Using Ollama with model: {self.model}")
    
    def create_system_prompt(self) -> str:
        """Create system prompt for the agent"""
        # Load system prompt from file
        prompt_file = Path(__file__).parent / "prompts" / "system_prompt.pmpt.tpl"
        return prompt_file.read_text()
    
    async def decide_action(self, task: str, state: BrowserState, history: List[str], trace_metadata: Optional[Dict[str, Any]] = None) -> AgentAction:
        """Decide next action based on current state"""
        
        # Build context
        history_str = "\n".join(history[-5:]) if history else "No previous actions"
        
        # Get the DOM representation that will be sent to LLM
        dom_representation = state.to_prompt_string()
        
        # Log the DOM that the AI sees
        logger.info("=== DOM SENT TO LLM ===")
        logger.info(dom_representation)
        logger.info("=== END DOM ===")
        
        # Optionally save to file
        if os.getenv("SAVE_DOM_TO_FILE", "false").lower() == "true":
            dom_file = Path(f"dom_snapshot_{int(time.time())}.txt")
            with open(dom_file, 'w') as f:
                f.write(f"Task: {task}\n\n")
                f.write(dom_representation)
                f.write(f"\n\nPrevious Actions:\n{history_str}")
            logger.info(f"DOM snapshot saved to {dom_file}")
        
        # Load user prompt template from file
        template_file = Path(__file__).parent / "prompts" / "user_prompt_template.md"
        template = template_file.read_text()
        prompt = template.format(
            task=task,
            dom_representation=dom_representation,
            history_str=history_str
        )
        
        try:
            if self.provider == "groq":
                # Call Groq API
                messages = [
                    {"role": "system", "content": self.create_system_prompt()},
                    {"role": "user", "content": prompt}
                ]
                
                # Prepare metadata for Langfuse tracing
                metadata = {
                    "task": task[:100],  # Truncate for metadata
                    "current_url": state.url,
                    "history_length": len(history),
                    "dom_elements_count": len(state.elements)
                }
                if trace_metadata:
                    metadata.update(trace_metadata)
                
                if self.using_langfuse:
                    # Use Langfuse-wrapped client with tracing
                    completion = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=0.7,
                        max_tokens=1024,
                        top_p=1,
                        response_format={"type": "json_object"},
                        name="browser_action_decision",
                        metadata=metadata
                    )
                else:
                    # Fallback to regular Groq client
                    completion = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=0.7,
                        max_completion_tokens=1024,
                        top_p=1,
                        response_format={"type": "json_object"}
                    )
                
                # Extract response
                result = json.loads(completion.choices[0].message.content)
                
            else:
                # Call ollama (existing code)
                response = self.client.chat(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.create_system_prompt()},
                        {"role": "user", "content": prompt}
                    ],
                    format="json",
                    options={"temperature": 0.7}
                )
                
                # Parse response
                result = json.loads(response['message']['content'])
            
            action = AgentAction(**result)
            
            logger.info(f"LLM decided: {action.action} - {action.reasoning[:100]}...")
            return action
            
        except Exception as e:
            logger.error(f"LLM decision failed: {e}")
            # Fallback action
            return AgentAction(
                reasoning="Error in LLM response, waiting",
                action=ActionType.WAIT
            )

# ============================================================================
# Main Agent
# ============================================================================

class BrowserAgent:
    """Main browser automation agent"""
    
    def __init__(self, task: str, enable_detailed_logging: bool = False, log_base_dir: Optional[str] = None, 
                 trace_id: Optional[str] = None, session_id: Optional[str] = None, user_id: Optional[str] = None):
        self.task = task
        self.browser = BrowserController()
        self.llm = LLMReasoner()
        self.history: List[str] = []
        self.step_count = 0
        self.enable_detailed_logging = enable_detailed_logging
        self.log_base_dir = Path(log_base_dir) if log_base_dir else None
        self.log_file = None
        
        # Langfuse trace metadata
        self.trace_id = trace_id
        self.session_id = session_id
        self.user_id = user_id
        self.trace = bool(trace_id)  # Flag to indicate if tracing is enabled
        
        # Create log file if detailed logging is enabled (fallback)
        if self.enable_detailed_logging and self.log_base_dir:
            self.log_base_dir.mkdir(parents=True, exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            self.log_file = self.log_base_dir / f"task_{timestamp}.log"
            self._log_detailed(f"Task initialized: {task}")
    
    def _log_detailed(self, message: str, metadata: Optional[Dict[str, Any]] = None):
        """Write to detailed log file"""
        if self.log_file:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            with open(self.log_file, 'a') as f:
                f.write(f"[{timestamp}] {message}\n")
    
    async def run(self, start_url: Optional[str] = None) -> Dict[str, Any]:
        """Run the agent to complete the task"""
        
        logger.info(f"Starting task: {self.task}")
        self._log_detailed(f"Starting task execution with URL: {start_url}")
        
        # Start browser
        await self.browser.start()
        
        # Navigate to start URL if provided
        if start_url:
            await self.browser.navigate(start_url)
        else:
            # Default to Google
            await self.browser.navigate("https://www.google.com")
        
        result = {
            "success": False,
            "steps": 0,
            "history": [],
            "final_url": "",
            "extracted_data": None
        }
        
        try:
            # Main agent loop
            while self.step_count < MAX_STEPS:
                self.step_count += 1
                logger.info(f"\n=== Step {self.step_count} ===")
                
                # Get current state
                state = await self.browser.get_state()
                logger.debug(f"Current URL: {state.url}")
                self._log_detailed(f"Step {self.step_count} - Current URL: {state.url}, Title: {state.title[:50] if state.title else 'N/A'}..., Elements: {len(state.elements)}")
                
                # Decide action
                trace_metadata = {
                    "langfuse_session_id": self.session_id,
                    "langfuse_user_id": self.user_id,
                    "step_count": self.step_count
                } if self.trace_id else None
                
                action = await self.llm.decide_action(self.task, state, self.history, trace_metadata)
                self._log_detailed(
                    f"Step {self.step_count} - Action decided: {action.action}, Reasoning: {action.reasoning[:100] if action.reasoning else 'N/A'}...",
                    metadata={"action": action.action, "step": self.step_count}
                )
                
                # Execute action
                success = await self.execute_action(action, state)
                
                # Record history
                history_entry = f"Step {self.step_count}: {action.action}"
                if action.element_index:
                    history_entry += f" on element [{action.element_index}]"
                if action.value:
                    history_entry += f" with value: {action.value[:50]}"
                history_entry += f" - {'Success' if success else 'Failed'}"
                self.history.append(history_entry)
                self._log_detailed(f"Step {self.step_count} - Execution result: {history_entry}")
                
                # Check if done
                if action.action == ActionType.DONE:
                    logger.info("Task completed!")
                    self._log_detailed(f"Task completed successfully. Extracted data: {action.value[:200] if action.value else 'None'}...")
                    result["success"] = True
                    result["extracted_data"] = action.value
                    break
                
                # Wait between actions
                await asyncio.sleep(THINK_TIME)
            
            # Prepare result
            result["steps"] = self.step_count
            result["history"] = self.history
            result["final_url"] = self.browser.page.url if self.browser.page else ""
            
        except Exception as e:
            logger.error(f"Agent error: {e}")
            self._log_detailed(f"Agent error occurred: {e}")
            result["error"] = str(e)
        
        finally:
            # Always close browser
            await self.browser.close()
            self._log_detailed(
                f"Task finished. Final state - Steps: {self.step_count}, Success: {result['success']}, Final URL: {result.get('final_url', 'N/A')}",
                metadata={"final_steps": self.step_count, "success": result['success']}
            )
        
        return result
    
    async def execute_action(self, action: AgentAction, state: BrowserState) -> bool:
        """Execute the decided action"""
        
        try:
            if action.action == ActionType.NAVIGATE:
                if action.value:
                    return await self.browser.navigate(action.value)
                return False
            
            elif action.action == ActionType.CLICK:
                if action.element_index:
                    return await self.browser.click_element(action.element_index)
                return False
            
            elif action.action == ActionType.TYPE:
                if action.element_index and action.value:
                    return await self.browser.type_text(action.element_index, action.value)
                return False
            
            elif action.action == ActionType.SCROLL:
                direction = action.value if action.value in ["up", "down"] else "down"
                return await self.browser.scroll(direction)
            
            elif action.action == ActionType.EXTRACT:
                text = await self.browser.extract_text()
                action.value = text[:1000]  # Limit extracted text
                return True
            
            elif action.action == ActionType.WAIT:
                await asyncio.sleep(2)
                return True
            
            elif action.action == ActionType.DONE:
                return True
            
            else:
                logger.warning(f"Unknown action: {action.action}")
                return False
                
        except Exception as e:
            logger.error(f"Action execution failed: {e}")
            return False

# ============================================================================
# Main Entry Point
# ============================================================================

async def main():
    """Example usage"""
    
    # Example tasks
    tasks = [
        ("Find the pricing for the gpt oss 120b 128k on groq", "https://www.qroq.com"),

        # ("Find Python documentation", "https://www.python.org"),
        # ("Search for flights from NYC to SF", "https://www.google.com/travel/flights"),
    ]
    
    for task_description, start_url in tasks:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running task: {task_description}")
        logger.info(f"{'='*60}")
        
        agent = BrowserAgent(task=task_description)
        result = await agent.run(start_url=start_url)
        
        # Print resultsX``
        logger.info("\n=== Results ===")
        logger.info(f"Success: {result['success']}")
        logger.info(f"Steps taken: {result['steps']}")
        logger.info(f"Final URL: {result['final_url']}")
        
        if result.get('extracted_data'):
            logger.info(f"Extracted: {result['extracted_data'][:200]}...")
        
        # Save results
        output_file = Path(f"results_{int(time.time())}.json")
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        logger.info(f"Results saved to {output_file}")

if __name__ == "__main__":
    # Check LLM provider availability
    if LLM_PROVIDER == "groq":
        if not GROQ_API_KEY:
            logger.error("GROQ_API_KEY not found in environment variables")
            logger.error("Please set GROQ_API_KEY in your .env file")
            sys.exit(1)
        logger.info(f"Using Groq provider with model: {GROQ_MODEL_MAP.get(MODEL_NAME, MODEL_NAME)}")
    else:
        # Check if ollama is available
        try:
            client = ollama.Client(host=OLLAMA_HOST)
            models = client.list()
            logger.info(f"Available models: {[m.model for m in models.models]}")
            
            # Check if our model is available
            model_names = [m.model for m in models.models]
            if not any(MODEL_NAME in name for name in model_names):
                logger.warning(f"Model {MODEL_NAME} not found. Please run: ollama pull {MODEL_NAME}")
                sys.exit(1)
        except Exception as e:
            logger.error(f"Cannot connect to ollama: {e}")
            logger.error("Please ensure ollama is running: ollama serve")
            sys.exit(1)
    
    # Run the agent
    asyncio.run(main())