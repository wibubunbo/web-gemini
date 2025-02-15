import logging
from pathlib import Path
from typing import Optional, Literal, Generator
import time
import base64
import io
import vertexai
from google.auth import default, transport
import openai
from PIL import Image
import numpy as np
from gradio import ChatMessage
import re
from io import BytesIO

import playwright.sync_api

from action.highlevel import HighLevelActionSet
from action.base import execute_python_code

from browser.observation import (
    extract_dom_snapshot,
    extract_merged_axtree,
    extract_screenshot,
    _pre_extract,
    _post_extract,
    extract_focused_element_bid,
    extract_dom_extra_properties,
)
from browser.constants import BID_ATTR, EXTRACT_OBS_MAX_TRIES
from browser import _get_global_playwright
from browser.utils import flatten_axtree_to_str, flatten_dom_to_str, prune_html


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def image_to_jpg_base64_url(image: np.ndarray | Image.Image):
    """Convert a numpy array to a base64 encoded image url."""
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    if image.mode in ("RGBA", "LA"):
        image = image.convert("RGB")

    with io.BytesIO() as buffer:
        image.save(buffer, format="JPEG")
        image_base64 = base64.b64encode(buffer.getvalue()).decode()

    return f"data:image/jpeg;base64,{image_base64}"

class WebAgent:
    """An intelligent web agent that can understand natural language instructions."""
    
    def __init__(
        self,
        # LLM settings
        model_name: str = "google/gemini-2.0-flash-001",
        project_id: str = "example-project",
        location: str = "us-central1",
        # Browser settings
        headless: bool = False,
        viewport: dict = {"width": 1280, "height": 720},
        slow_mo: int = 1000,  # ms
        timeout: int = 30000,  # ms
        locale: Optional[str] = None,
        timezone_id: Optional[str] = None,
        tags_to_mark: Literal["all", "standard_html"] = "standard_html",
        demo_mode: Literal["off", "default", "all_blue", "only_visible_elements"] = "default",
        # Observation settings
        use_html: bool = True,
        use_axtree: bool = True,
        use_screenshot: bool = False,
    ):
        """Initialize the web agent."""
        # Browser settings
        self.headless = headless
        self.viewport = viewport
        self.slow_mo = slow_mo
        self.timeout = timeout
        self.locale = locale
        self.timezone_id = timezone_id
        self.tags_to_mark = tags_to_mark
        self.demo_mode = demo_mode
        
        # LLM settings
        self.model_name = model_name
        self.use_html = use_html
        self.use_axtree = use_axtree
        self.use_screenshot = use_screenshot

        if not (use_html or use_axtree):
            raise ValueError(f"Either use_html or use_axtree must be set to True.")

        # Initialize Vertex AI
        vertexai.init(project=project_id, location=location)
        credentials, _ = default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        auth_request = transport.requests.Request()
        credentials.refresh(auth_request)

        # Initialize OpenAI client with Vertex AI endpoint
        self.openai_client = openai.OpenAI(
            base_url=f"https://{location}-aiplatform.googleapis.com/v1beta1/projects/{project_id}/locations/{location}/endpoints/openapi",
            api_key=credentials.token,
        )
        
        # Configure action system
        self.action_set = HighLevelActionSet(
            subsets=["chat", "tab", "nav", "bid", "infeas"],
            strict=False,
            multiaction=False,
            demo_mode=demo_mode
        )

        # Initialize Playwright components
        self.browser: playwright.sync_api.Browser = None
        self.context: playwright.sync_api.BrowserContext = None
        self.page: playwright.sync_api.Page = None
        self.page_history: dict = {}

        # Initialize chat and state variables
        self.messages: list[dict] = []
        self.action_history: list[str] = []
        self.last_error: str = ""
        self.last_action: str = ""
        self.infeasible_message_received: bool = False

        # Gradio components
        self.user_msg: str = None
        self.url_input: str = None
        self.image_display: np.ndarray = None
        self.history: list[ChatMessage] = []

    def handle_url(self, url: str) -> tuple:
        """Handle URL navigation"""
        self.start(url)
        screenshot = extract_screenshot(self.page)
        return url, screenshot
    
    def set_user_msg(self, user_msg: str) -> tuple:
        self.history.append(ChatMessage(role="user", content=user_msg))
        self.messages.append({"role": "user", "content": user_msg})
        return "", self.history        

    def process_step(self, url: str, image: np.ndarray, history: list[ChatMessage]) -> Generator:
        """Process instruction step by step"""
        self.url_input = url
        self.image_display = image
        self.history = history
        while True:

            # Add initial thinking message
            self.history.append(ChatMessage(
                role="assistant",
                content=f"🤔 ..."
            ))
            yield self._get_gradio_state()
        
            obs = self._get_observation()  
            is_complete, reasoning, action = self._get_llm_response(obs)
            # If goal is complete, stop
            if is_complete:
                completion_msg = "Objective reached. Stopping..."
                logger.info(completion_msg)
                self.history[-1] = ChatMessage(
                    role="assistant",
                    content=f"🥳 The objective was successfully executed.",
                )
                yield self._get_gradio_state()
                break

            # If agent returns None action, stop
            if not action:
                logger.debug("Agent returned None action.")
                self.history.append(ChatMessage(role="assistant", content="❌ Agent returned None action."))
                yield self._get_gradio_state()
                break
                
            # Store action
            self.last_action = reasoning + "\n" + action
            
            # Add action to history
            logger.info(self.last_action)

            if action.startswith("send_msg_to_user") or action.startswith("report_infeasible_instructions"):
                self.history[-1] = ChatMessage(
                    role="assistant",
                    content=f"🤔 {reasoning}",
                )

                self.history.append(ChatMessage(
                    role="assistant",
                    content=f"⏳ Action: {action.split('(')[0].strip()}",
                ))
                
            else:
                self.history[-1] = ChatMessage(
                    role="assistant",
                    content=f"🤔 {reasoning}",
                )
                self.history.append(ChatMessage(
                    role="assistant",
                    content=f"⏳ Action: {action}",
                ))

            yield self._get_gradio_state()

            # Execute action
            try:
                code = self.action_set.to_python_code(reasoning + "\n" + "```" + action + "```")
                
                def send_message_to_user(text: str):
                    if not isinstance(text, str):
                        raise ValueError(f"Message must be a string, got {text}")
                    logger.info(text)
                    self.messages.append({"role": "assistant", "content": text})
                    self.history.append(ChatMessage(role="assistant", content=text))
                        
                def report_infeasible_instructions(reason: str):
                    if not isinstance(reason, str):
                        raise ValueError(f"Reason must be a string, got {reason}")
                    logger.info(reason)
                    self.messages.append({"role": "infeasible", "content": reason})
                    self.history.append(ChatMessage(role="assistant", content=reason))
                    self.infeasible_message_received = True

                execute_python_code(
                    code,
                    self.page,
                    send_message_to_user=send_message_to_user,
                    report_infeasible_instructions=report_infeasible_instructions,
                )

                self.last_error = ""

                if self.messages[-1]["role"] == "assistant" or self.messages[-1]["role"] == "infeasible":
                    self.history[-2] = ChatMessage(
                        role="assistant",
                        content=f"✅ Action: {action.split('(')[0].strip()}",
                    )
                else:
                    self.history[-1] = ChatMessage(
                        role="assistant",
                        content=f"✅ Action: {action}",
                    )                    
                    
            except Exception as e:
                error_msg = f"{type(e).__name__}: {e}"
                logger.info(error_msg)
                self.last_error = error_msg
                if self.messages[-1]["role"] == "assistant" or self.messages[-1]["role"] == "infeasible":
                    self.history[-2] = ChatMessage(
                        role="assistant",
                        content=f"❌ Action: {action.split('(')[0].strip()}",
                    )
                else:
                    self.history[-1] = ChatMessage(
                        role="assistant",
                        content=f"❌ Action: {action}",
                    )  
    
                self.history.append(ChatMessage(
                    role="assistant",
                    content=f"❌ {error_msg}",
                ))
                    
            # Wait for page to stabilize
            time.sleep(0.5)
            self.context.cookies()
            self._wait_dom_loaded()
            self._active_page_check()                
            yield self._get_gradio_state()

            # Check if infeasible
            if self.infeasible_message_received or action.startswith("send_msg_to_user"):
                break
            
    def clear_chat(self) -> tuple:
        self.history = []
        self.messages = []
        self.action_history = []
        self.last_error = ""
        self.last_action = ""
        self.infeasible_message_received = False
        return "", "", self.history, None

    def _get_gradio_state(self) -> tuple:
        """Get current state for Gradio UI updates"""
        # Update screenshot
        screenshot = extract_screenshot(self.page)
        return (
            self.page.url,
            screenshot,
            self.history,
        )

    def _get_llm_response(self, obs: dict) -> dict:
        """Get next action from LLM based on current observation."""
        system_msgs = [{
            "type": "text",
            "text": """# Instructions
            You are an advanced Web UI Assistant designed to help users perform tasks in a web browser. Your core responsibilities are:
            1. Maintain context across multiple interactions
            2. Precisely track goal completion status
            3. Communicate clearly with users
            4. Execute appropriate web actions

            # Key Principles
            1. NEVER mark a task as complete (is_complete=true) unless you have:
            - Fully achieved the user's stated goal
            - Communicated relevant results to the user
            - Verified success through page state
            
            2. When handling user queries:
            - Break down complex tasks into steps
            - Verify each step's completion
            - Keep users informed of progress
            - Handle errors gracefully

            3. Communication Guidelines:
            - Use send_msg_to_user() to provide updates
            - Report obstacles with report_infeasible_instructions()
            - Explain your reasoning clearly
            - Ask for clarification when needed

            # Response Format
            You must respond with a JSON object in this exact format:
            {
                "goal_status": {
                    "is_complete": boolean,
                    "reason": "detailed explanation of current status"
                },
                "next_action": {
                    "reasoning": "step-by-step thought process",
                    "action": "action code"
                }
            }

            # Response Examples
            Example 1 - Task in Progress:
            {
                "goal_status": {
                    "is_complete": false,
                    "reason": "Still need to fill out the contact form and submit it"
                },
                "next_action": {
                    "reasoning": "1. Located the contact form\n2. Next step is to fill in the name field\n3. Will use type action on the input with bid 15",
                    "action": "type(\\"15\\", \\"John Doe\\")"
                }
            }

            Example 2 - Communicating with User:
            {
                "goal_status": {
                    "is_complete": false,
                    "reason": "Need to inform user about required information"
                },
                "next_action": {
                    "reasoning": "The form requires a phone number but none was provided. Need to ask user.",
                    "action": "send_msg_to_user(\\"I see that a phone number is required. Could you please provide your phone number?\\")"
                }
            }

            Example 3 - Task Completion:
            {
                "goal_status": {
                    "is_complete": true,
                    "reason": "Successfully purchased the item and sent order confirmation to user"
                }
            }

            Example 4 - Handling Infeasible Tasks:
            {
                "goal_status": {
                    "is_complete": false,
                    "reason": "Cannot proceed due to technical limitations"
                },
                "next_action": {
                    "reasoning": "The requested action requires JavaScript execution which is not supported",
                    "action": "report_infeasible_instructions(\\"I apologize, but I cannot execute custom JavaScript code. Is there another way I can help you achieve your goal?\\")"
                }
            }
            """
        }]

        user_msgs = []
        
        # Add chat messages
        user_msgs.append({
            "type": "text",
            "text": "# Chat Messages"
        })
        
        for msg in obs["chat_messages"]:
            if msg["role"] in ("user", "assistant", "infeasible"):
                user_msgs.append({
                    "type": "text",
                    "text": f"- [{msg['role']}] {msg['content']}"
                })
            elif msg["role"] == "user_image":
                user_msgs.append({"type": "image_url", "image_url": msg["message"]})
            else:
                raise ValueError(f"Unexpected chat message role {repr(msg['role'])}")

        # Add page URLs
        user_msgs.append({
            "type": "text", 
            "text": "\n# Currently open tabs\n\n"
        })
        
        for page_index, (page_url, page_title) in enumerate(zip(obs["open_pages_urls"], obs["open_pages_titles"])):
            user_msgs.append({
                "type": "text",
                "text": f"""Tab {page_index}{" (active tab)" if page_index == obs["active_page_index"] else ""}\n Title: {page_title}\n URL: {page_url}"""
            })

        # Add accessibility tree
        if self.use_axtree:
            user_msgs.append({
                "type": "text",
                "text": f"\n# Current page Accessibility Tree\n\n{obs['axtree_txt']}\n"
            })

        # Add HTML
        if self.use_html:
            user_msgs.append({
                "type": "text", 
                "text": f"\n# Current page DOM\n\n{obs['pruned_html']}\n"
            })

        # Add screenshot
        if self.use_screenshot:
            user_msgs.append({
                "type": "text",
                "text": "\n# Current page Screenshot\n\n"
            })
            user_msgs.append({
                "type": "image_url",
                "image_url": {
                    "url": image_to_jpg_base64_url(obs["screenshot"]),
                    "detail": "auto"
                }
            })

        # Add action space description
        user_msgs.append({
            "type": "text",
            "text": f"""\n# Action Space\n\n{self.action_set.describe(with_long_description=False, with_examples=True)}"""
        })

        # Add action history and errors
        if self.action_history:
            user_msgs.append({
                "type": "text",
                "text": "\n# History of past actions\n"
            })
            user_msgs.extend([{
                "type": "text",
                "text": f"\n{action}"
            } for action in self.action_history])

            if obs["last_action_error"]:
                user_msgs.append({
                    "type": "text",
                    "text": f"\n# Error message from last action\n\n{self.last_error}"
                })

        # Get LLM response
        response = self.openai_client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_msgs},
                {"role": "user", "content": user_msgs}
            ],
            max_tokens=8000,
            temperature=0.3,
            response_format={"type": "json_object"}  # Ensure JSON response
        )

        logger.info(f"LLM response: {response.choices[0].message.content}")
        
        try:
            import json
            result = json.loads(response.choices[0].message.content)
            
            is_complete = result["goal_status"]["is_complete"]
            
            reasoning = None
            action = None
            if not is_complete and "next_action" in result:
                reasoning = result["next_action"]["reasoning"]
                action = result["next_action"]["action"]
                # Store full response for history
                self.action_history.append(
                    f"{reasoning}\n"
                    f"{action}"
                )
            
            return is_complete, reasoning, action
            
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Error parsing LLM response: {e}")
            logger.error(f"Raw response: {response.choices[0].message.content}")
            return False, None, None
    
    
    def start(self, start_url: str):
        """Start the agent and navigate to initial URL."""
        # Launch browser if needed
        if not self.browser:
            pw = _get_global_playwright()
            self.browser = pw.chromium.launch(
                headless=self.headless,
                slow_mo=self.slow_mo,
                args=[f"--window-size={self.viewport['width']},{self.viewport['height']}"],
            )

        # Set up browser context
        if not self.context:
            self.context = self.browser.new_context(
                viewport=self.viewport,
                locale=self.locale,
                timezone_id=self.timezone_id,
            )
            self.context.set_default_timeout(self.timeout)

            # Set up bid attribute
            pw = _get_global_playwright()
            pw.selectors.set_test_id_attribute(BID_ATTR)

            # Set up page activation tracking
            self.context.expose_binding(
                "browsergym_page_activated",
                lambda source: self._activate_page_from_js(source["page"])
            )
            self.context.add_init_script(
                r"""
                window.browsergym_page_activated();
                window.addEventListener("focus", () => {window.browsergym_page_activated();}, {capture: true});
                window.addEventListener("focusin", () => {window.browsergym_page_activated();}, {capture: true});
                window.addEventListener("load", () => {window.browsergym_page_activated();}, {capture: true});
                window.addEventListener("pageshow", () => {window.browsergym_page_activated();}, {capture: true});
                window.addEventListener("mousemove", () => {window.browsergym_page_activated();}, {capture: true});
                window.addEventListener("mouseup", () => {window.browsergym_page_activated();}, {capture: true});
                window.addEventListener("mousedown", () => {window.browsergym_page_activated();}, {capture: true});
                window.addEventListener("wheel", () => {window.browsergym_page_activated();}, {capture: true});
                window.addEventListener("keyup", () => {window.browsergym_page_activated();}, {capture: true});
                window.addEventListener("keydown", () => {window.browsergym_page_activated();}, {capture: true});
                window.addEventListener("input", () => {window.browsergym_page_activated();}, {capture: true});
                window.addEventListener("touchstart", () => {window.browsergym_page_activated();}, {capture: true});
                window.addEventListener("touchend", () => {window.browsergym_page_activated();}, {capture: true});
                document.addEventListener("visibilitychange", () => {
                    if (document.visibilityState === "visible") {
                        window.browsergym_page_activated();
                    }
                }, {capture: true});
                """
            )

        # Create initial page
        if not self.page:
            self.page = self.context.new_page()

        # Send welcome message
        self.messages.append({"role": "assistant", "content": "Hi! I'm your web assistant. I can help you interact with websites. What would you like me to do?"})
        
        # Navigate to start URL
        self.page.goto(start_url)
        self._wait_dom_loaded()
        self._active_page_check()


    def _get_observation(self) -> dict:
        """Get the current observation of the web page state."""
        
        for retries_left in reversed(range(EXTRACT_OBS_MAX_TRIES)):
            try:
                # Pre-extraction setup
                _pre_extract(self.page, tags_to_mark=self.tags_to_mark, lenient=(retries_left == 0))

                # Extract page state
                dom = extract_dom_snapshot(self.page)
                axtree = extract_merged_axtree(self.page)
                focused_element_bid = extract_focused_element_bid(self.page)
                extra_properties = extract_dom_extra_properties(dom)

            except (playwright.sync_api.Error) as e:
                if retries_left > 0:
                    logger.warning(
                        f"Error extracting observation, retrying ({retries_left} tries left)\n{e}"
                    )
                    _post_extract(self.page)
                    time.sleep(0.5)
                    continue
                else:
                    raise e
            break

        # Post-extraction cleanup
        _post_extract(self.page)

        # Build observation dict
        obs = {
            "chat_messages": tuple(self.messages),
            "open_pages_urls": [p.url for p in self.context.pages],
            "open_pages_titles": [p.title() for p in self.context.pages],
            "active_page_index": self.context.pages.index(self.page),
            "url": self.page.url,
            "screenshot": extract_screenshot(self.page),
            "dom_object": dom,
            "axtree_object": axtree,
            "extra_element_properties": extra_properties,
            "focused_element_bid": focused_element_bid,
            "axtree_txt": flatten_axtree_to_str(axtree),
            "pruned_html": prune_html(flatten_dom_to_str(dom)),
            "last_action": self.last_action,
            "last_action_error": self.last_error,
        }

        return obs
    

    def _activate_page_from_js(self, page: playwright.sync_api.Page):
        """Handle page activation from JavaScript."""
        if page.context != self.context:
            raise RuntimeError(f"Page {page} belongs to different context")

        if page in self.page_history:
            self.page_history[page] = self.page_history.pop(page)
        else:
            self.page_history[page] = None

        self.page = page


    def _active_page_check(self):
        """Verify active page is valid and create new one if needed."""
        if len(self.context.pages) == 0:
            logger.warning("No pages open, creating new page")
            self.page = self.context.new_page()

        while self.page_history and (self.page.is_closed() or self.page not in self.context.pages):
            self.page_history.pop(self.page)
            self.page = list(self.page_history.keys())[-1]

        if self.page not in self.context.pages:
            raise RuntimeError(f"Active page not in context: {self.page}")

        if self.page.is_closed():
            raise RuntimeError(f"Active page is closed: {self.page}")


    def _wait_dom_loaded(self):
        """Wait for DOM content to be loaded in all pages/frames."""
        for page in self.context.pages:
            try:
                page.wait_for_load_state("domcontentloaded", timeout=3000)
            except playwright.sync_api.Error:
                pass
            for frame in page.frames:
                try:
                    frame.wait_for_load_state("domcontentloaded", timeout=3000)
                except playwright.sync_api.Error:
                    pass


    def close(self):
        """Clean up resources."""
        if self.context:
            self.context.close()
            self.context = None
            
        if self.browser:
            self.browser.close()
            self.browser = None
