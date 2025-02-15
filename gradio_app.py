import gradio as gr
import argparse
from typing import Optional, Dict
from agent.web_agent import WebAgent

class GradioWebAgentDemo:
    """Gradio UI for WebAgent"""
    
    def __init__(self):
        self.agent: Optional[WebAgent] = None
        self.model_choices: Dict[str, str] = {
            "Gemini 2.0 Flash": "google/gemini-2.0-flash-001",
            "Gemini 1.5 Pro": "google/gemini-1.5-pro"
        }

    def initialize_agent(self, model, project_id, location, headless, use_html, use_axtree, use_screenshot):
        """Initialize WebAgent with the given configuration"""
        if not project_id or not location:
            return gr.Markdown("❌ Please provide valid Google Cloud Project ID and Location")
        
        try:
            self.agent = WebAgent(
                model_name=self.model_choices[model],
                project_id=project_id,
                location=location,
                headless=headless,
                use_html=use_html,
                use_axtree=use_axtree,
                use_screenshot=use_screenshot
            )
            return gr.Markdown("✅ Agent initialized successfully! You can start browsing.")
        except Exception as e:
            return gr.Markdown(f"❌ Failed to initialize agent: {str(e)}")
        
    def handle_url(self, url):
        """Wrapper to handle URL navigation with agent check"""
        if self.agent is None:
            raise ValueError("Agent not initialized. Please initialize the agent first.")
        return self.agent.handle_url(url)

    def set_user_msg(self, msg):
        """Wrapper to handle user messages with agent check"""
        if self.agent is None:
            raise ValueError("Agent not initialized. Please initialize the agent first.")
        return self.agent.set_user_msg(msg)
        
    def process_step(self, url, image, chatbot):
        """Wrapper to handle processing with agent check"""
        if self.agent is None:
            raise ValueError("Agent not initialized. Please initialize the agent first.")            
        # Get the generator from process_step
        for state in self.agent.process_step(url, image, chatbot):
            yield state
    
    def clear_chat(self):
        """Wrapper to handle chat clearing with agent check"""
        if self.agent is None:
            raise ValueError("Agent not initialized. Please initialize the agent first.")
        return self.agent.clear_chat()
        
    def launch(self, share=True):
        with gr.Blocks(theme='NoCrypt/miku') as demo:
            with gr.Row():
                gr.HTML("""
                <style>
                    .header-container {
                        text-align: center;
                        padding: 2rem 1rem;
                        background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
                        border-radius: 16px;
                        margin-bottom: 1.5rem;
                        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
                    }
                    .header-title {
                        margin: 0;
                        color: white;
                        font-size: 2.8em;
                        font-weight: 700;
                        letter-spacing: -0.025em;
                        line-height: 1.2;
                    }
                    .header-subtitle {
                        color: rgba(255, 255, 255, 0.9);
                        font-size: 1.2em;
                        margin: 0.75rem 0;
                        font-weight: 400;
                    }
                </style>
                <div class="header-container">
                    <h1 class="header-title">🤖 WebGemini </h1>
                    <p class="header-subtitle">Your AI-powered companion for seamless web navigation and interaction</p>
                </div>
            """)

            with gr.Accordion("Configuration", open=True):
                with gr.Row():
                    project_id_input = gr.Textbox(
                        label="Project ID", placeholder="Enter your Google Cloud Project ID"
                    )
                    location_input = gr.Textbox(
                        label="Location",
                        placeholder="Enter your Google Cloud Location (e.g., us-central1)",
                    )
                
                with gr.Row():
                    model_dropdown = gr.Dropdown(
                        label="Select Model",
                        choices=list(self.model_choices.keys()),  # Use keys for display
                        value="Gemini 2.0 Flash",  # Default selection
                        interactive=True
                    )

                with gr.Row():
                    headless_checkbox = gr.Checkbox(label="Headless", value=False)
                    use_html_checkbox = gr.Checkbox(label="Use HTML", value=True)
                    use_axtree_checkbox = gr.Checkbox(label="Use AXTree", value=True)
                    use_screenshot_checkbox = gr.Checkbox(label="Use Screenshot", value=False)

                init_button = gr.Button("Initialize Agent")
                status = gr.Markdown("Configure the agent and click Initialize Agent button to begin")

            with gr.Row():
                with gr.Column():
                    url_input = gr.Textbox(
                        label="🔗 Start URL",
                        placeholder="Enter starting URL to navigate to (e.g. https://www.google.com) and press Enter ...",
                        show_label=True,
                        submit_btn=True
                    )
                    
            with gr.Row():
                with gr.Column(scale=8):
                    image_display = gr.Image(
                        label="🖥️ Browser View",
                        show_label=True,
                        interactive=False,
                        height=600,
                        show_download_button=True
                    )
                with gr.Column(scale=2):
                    chatbot = gr.Chatbot(
                        label="💬 Agent Output",
                        show_label=True,
                        type="messages",
                        height=600 ,
                        layout="bubble"                 
                    )

            with gr.Row(equal_height=True):
                with gr.Column(scale=4):
                    chat_input = gr.Textbox(
                        label="💭 Chat with the agent",
                        show_label=True,
                        placeholder="Type your message and press Enter..."                        
                    )
                with gr.Column(scale=1):
                    clear_btn = gr.Button(
                        "Clear Chat"
                    )

            # Initialize WebAgent
            init_button.click(
                self.initialize_agent,
                inputs=[
                    model_dropdown,
                    project_id_input,
                    location_input,
                    headless_checkbox,
                    use_html_checkbox,
                    use_axtree_checkbox,
                    use_screenshot_checkbox
                ],
                outputs=[status]
            )
                    
            # Handle URL navigation
            url_input.submit(
                fn=self.handle_url,
                inputs=[url_input],
                outputs=[url_input, image_display]
            )

            # Handle chat input
            chat_input.submit(
                fn=self.set_user_msg,
                inputs=[chat_input],
                outputs=[chat_input, chatbot]
            ).then(
                fn=self.process_step,
                inputs=[
                    url_input,
                    image_display,
                    chatbot
                ],
                outputs=[
                    url_input,
                    image_display,
                    chatbot
                ]
            )

            # Handle clear chat
            clear_btn.click(
                fn=self.clear_chat,
                inputs=None,
                outputs=[url_input, chat_input, chatbot, image_display]
            )
            
        demo.queue().launch(share=share)

if __name__ == "__main__":
    # Initialize WebAgent with command line arguments
    ui = GradioWebAgentDemo()
    ui.launch()