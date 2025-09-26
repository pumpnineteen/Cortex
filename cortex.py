import sys
import json
import asyncio
import aiohttp
import os
from pathlib import Path
import yaml
from datetime import datetime
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QTextEdit,
    QLineEdit,
    QPushButton,
    QComboBox,
    QLabel,
    QSplitter,
    QProgressBar,
    QTabWidget,
    QFrame,
    QGroupBox,
    QGridLayout,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QDialog,
    QDialogButtonBox,
    QScrollArea,
)
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QFont, QTextCursor, QPalette
from PySide6.QtWebEngineWidgets import QWebEngineView
import markdown
from markdown.extensions import codehilite, fenced_code, tables
import markdown.extensions.extra
import re
import traceback

from knowledge_cache import CachedIterativeChatManager
from query_analyzer import EnhancedLLMQueryAnalyzer
from reasoning_model_manager import ReasoningModelManager, ReasoningParser
from enhanced_markdown_display import EnhancedMarkdownDisplay
from enhanced_log_display import EnhancedLogDisplay
from config_manager import ConfigManager
import html

class OllamaStatusChecker(QThread):
    """Worker thread for checking Ollama status and fetching models"""

    status_changed = Signal(bool, str)
    models_updated = Signal(list)
    connection_lost = Signal()

    def __init__(self):
        super().__init__()
        self.running = True
        self.ollama_base_url = "http://localhost:11434"
        self.has_models = False
        self.poll_continuously = True

    async def check_status_and_models(self):
        """Check if Ollama is running and fetch available models"""
        try:
            async with aiohttp.ClientSession() as session:
                # Check if Ollama is running
                async with session.get(
                    f"{self.ollama_base_url}/api/version", timeout=aiohttp.ClientTimeout(total=3)
                ) as response:
                    if response.status == 200:
                        # Only fetch models if we don't have them or if forced
                        if not self.has_models or self.poll_continuously:
                            try:
                                async with session.get(f"{self.ollama_base_url}/api/tags") as models_response:
                                    if models_response.status == 200:
                                        data = await models_response.json()
                                        models = [model["name"] for model in data.get("models", [])]
                                        self.models_updated.emit(models)
                                        self.has_models = True
                                        self.poll_continuously = False  # Stop continuous polling
                                        self.status_changed.emit(True, f"Ollama running - {len(models)} models available")
                                    else:
                                        self.status_changed.emit(True, "Ollama running - models list unavailable")
                            except Exception:
                                self.status_changed.emit(True, "Ollama running - models list error")
                        else:
                            # Just confirm it's still running without fetching models
                            self.status_changed.emit(True, "Ollama running")
                    else:
                        self.has_models = False
                        self.poll_continuously = True
                        self.status_changed.emit(False, f"Ollama responded with status {response.status}")
        except Exception:
            self.has_models = False
            self.poll_continuously = True
            self.connection_lost.emit()
            self.status_changed.emit(False, "Ollama not responding")

    def force_model_refresh(self):
        """Force a model list refresh"""
        self.poll_continuously = True

    def run(self):
        """Run the status check loop"""
        while self.running:
            asyncio.run(self.check_status_and_models())
            # Longer interval when we have models and connection is stable
            sleep_time = 10000 if self.has_models and not self.poll_continuously else 5000
            self.msleep(sleep_time)

    def stop(self):
        """Stop the status checker"""
        self.running = False


class ModelConfigDialog(QDialog):
    """Dialog for configuring models for different tasks"""

    def __init__(self, config_manager, available_models, parent=None):
        super().__init__(parent)
        self.config_manager = config_manager
        self.available_models = available_models
        self.model_combos = {}
        self.setup_ui()

    def setup_ui(self):
        """Setup the dialog UI"""
        self.setWindowTitle("Model Configuration")
        self.setModal(True)
        self.resize(600, 400)

        layout = QVBoxLayout(self)

        # Scroll area for model configurations
        scroll = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)

        # Model tasks
        tasks = [
            ("chat", "Main Chat Model", "Primary model for general conversation"),
            ("search_extract", "Search Extraction", "Extract relevant info from web search results"),
            ("search_summarize", "Search Summarization", "Summarize extracted search information"),
            ("code_analyze", "Code Analysis", "Analyze and explain code functionality"),
            ("code_summarize", "Code Summarization", "Create hierarchical code summaries"),
            ("embedding", "Embedding Model", "Vector embeddings for RAG (typically nomic-embed-text)"),
        ]

        for task_key, task_name, task_desc in tasks:
            group = QGroupBox(task_name)
            group_layout = QVBoxLayout(group)

            # Description
            desc_label = QLabel(task_desc)
            desc_label.setStyleSheet("color: #666; font-style: italic;")
            desc_label.setWordWrap(True)
            group_layout.addWidget(desc_label)

            # Model selection
            combo = QComboBox()
            combo.addItem("-- No model selected --", "")

            current_model = self.config_manager.get(f"ollama.models.{task_key}", "")
            current_index = 0

            for i, model in enumerate(self.available_models, 1):
                combo.addItem(model, model)
                if model == current_model:
                    current_index = i

            combo.setCurrentIndex(current_index)
            self.model_combos[task_key] = combo

            group_layout.addWidget(combo)
            scroll_layout.addWidget(group)

        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll)

        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_selected_models(self):
        """Get the selected models"""
        return {task: combo.currentData() for task, combo in self.model_combos.items()}


class PromptConfigDialog(QDialog):
    """Dialog for configuring prompts for different tasks"""

    def __init__(self, config_manager, parent=None):
        super().__init__(parent)
        self.config_manager = config_manager
        self.prompt_editors = {}
        self.setup_ui()

    def setup_ui(self):
        """Setup the dialog UI"""
        self.setWindowTitle("Prompt Configuration")
        self.setModal(True)
        self.resize(800, 600)

        layout = QVBoxLayout(self)

        # Tab widget for different prompts
        tabs = QTabWidget()

        prompts = [
            ("chat", "Chat", "Main conversation prompt"),
            ("search_extract", "Search Extract", "Web search information extraction"),
            ("search_summarize", "Search Summarize", "Web search result summarization"),
            ("code_analyze", "Code Analysis", "Code analysis and explanation"),
            ("code_summarize", "Code Summary", "Codebase hierarchical summarization"),
        ]

        for prompt_key, tab_name, description in prompts:
            tab_widget = QWidget()
            tab_layout = QVBoxLayout(tab_widget)

            # Description
            desc_label = QLabel(description)
            desc_label.setStyleSheet("font-weight: bold; color: #333;")
            tab_layout.addWidget(desc_label)

            # Prompt editor
            editor = QTextEdit()
            editor.setPlainText(self.config_manager.get(f"prompts.{prompt_key}", ""))
            editor.setFont(QFont("Consolas", 10))
            self.prompt_editors[prompt_key] = editor
            tab_layout.addWidget(editor)

            # Available variables info
            if "{" in self.config_manager.get(f"prompts.{prompt_key}", ""):
                info_label = QLabel("Available variables: {query}, {content}, {code}")
                info_label.setStyleSheet("color: #666; font-style: italic;")
                tab_layout.addWidget(info_label)

            tabs.addTab(tab_widget, tab_name)

        layout.addWidget(tabs)

        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_prompts(self):
        """Get the edited prompts"""
        return {prompt: editor.toPlainText() for prompt, editor in self.prompt_editors.items()}
    
class OllamaWorker(QThread):
    """Worker thread for Ollama API communication"""

    response_received = Signal(str)
    error_occurred = Signal(str)
    stream_chunk = Signal(str)

    def __init__(self, model, message, stream=True, system_prompt=""):
        super().__init__()
        self.model = model
        self.message = message
        self.stream = stream
        self.system_prompt = system_prompt
        self.ollama_url = "http://localhost:11434/api/chat"

    async def send_request(self):
        """Send request to Ollama API"""
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": self.message})

        payload = {"model": self.model, "messages": messages, "stream": self.stream}

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.ollama_url, json=payload) as response:
                    if response.status == 200:
                        if self.stream:
                            full_response = ""
                            async for line in response.content:
                                if line:
                                    try:
                                        data = json.loads(line.decode("utf-8"))
                                        if "message" in data and "content" in data["message"]:
                                            chunk = data["message"]["content"]
                                            full_response += chunk
                                            self.stream_chunk.emit(chunk)
                                        if data.get("done", False):
                                            self.response_received.emit(full_response)
                                            break
                                    except json.JSONDecodeError:
                                        continue
                        else:
                            data = await response.json()
                            if "message" in data and "content" in data["message"]:
                                self.response_received.emit(data["message"]["content"])
                    else:
                        self.error_occurred.emit(f"HTTP {response.status}: {await response.text()}")
        except Exception as e:
            self.error_occurred.emit(f"Connection error: {str(e)}")

    def run(self):
        """Run the async request in thread"""
        asyncio.run(self.send_request())


class IterativeOllamaWorker(QThread):
    """Enhanced worker thread for iterative chat processing with detailed logging"""

    response_received = Signal(str)
    error_occurred = Signal(str)
    status_updated = Signal(str)
    analysis_completed = Signal(dict)
    log_entry = Signal(str, str, str)  # type, message, details

    def __init__(self, config_manager, user_query, conversation_context=None):
        super().__init__()
        self.config_manager = config_manager
        self.user_query = user_query
        self.conversation_context = conversation_context or []

    def _log_callback(self, log_type: str, message: str, details: str = ""):
        """Callback to emit log entries via Qt signals"""
        self.log_entry.emit(log_type, message, details)

    async def process_query(self):
        """Process the query through the aggressive search system with detailed logging"""
        try:
            # Create chat manager with caching and logging - pass the callback!
            chat_manager = CachedIterativeChatManager(
                self.config_manager,
                status_callback=self.status_updated.emit,
                log_callback=self._log_callback,  # Pass our callback method
            )

            # Process with aggressive search strategy
            result = await chat_manager.process_query(self.user_query, self.conversation_context)

            # Log cache statistics
            cached_searches = sum(1 for r in result.get("search_results", {}).values() if r.get("cached"))
            fresh_searches = len(result.get("search_results", {})) - cached_searches

            if cached_searches > 0:
                self.log_entry.emit("INFO", f"Used {cached_searches} cached results, {fresh_searches} fresh searches", "")

            # Emit results
            self.analysis_completed.emit(result)
            self.response_received.emit(result["response"])
            self.log_entry.emit(
                "SUCCESS", "Aggressive search processing completed", f"Response: {len(result['response'])} chars"
            )

        except Exception as e:
            full_traceback = traceback.format_exc()
            error_msg = f"Processing error: {str(e)}"
            self.log_entry.emit("ERROR", "Aggressive search processing failed", full_traceback)
            self.error_occurred.emit(error_msg)

    def run(self):
        """Run the async processing in thread"""
        asyncio.run(self.process_query())


class AnalysisDisplayWidget(QWidget):
    """Widget to display query analysis and detailed processing log"""

    def __init__(self):
        super().__init__()
        self.setup_ui()
        self.log_entries = []

    def setup_ui(self):
        layout = QVBoxLayout(self)

        # Analysis section (compact)
        analysis_group = QGroupBox("Query Analysis")
        analysis_layout = QVBoxLayout(analysis_group)

        # Create a horizontal layout for reasoning and confidence
        info_layout = QHBoxLayout()

        self.reasoning_label = QLabel("No analysis yet")
        self.reasoning_label.setWordWrap(True)
        self.reasoning_label.setStyleSheet("font-style: italic; color: #666;")
        info_layout.addWidget(self.reasoning_label, stretch=3)

        self.confidence_label = QLabel("Confidence: N/A")
        self.confidence_label.setStyleSheet("font-weight: bold;")
        self.confidence_label.setMinimumWidth(120)
        info_layout.addWidget(self.confidence_label, stretch=1)

        analysis_layout.addLayout(info_layout)
        layout.addWidget(analysis_group)

        # Detailed processing log (main area)
        log_group = QGroupBox("Processing Log")
        log_layout = QVBoxLayout(log_group)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Consolas", 9))
        self.log_text.setStyleSheet("""
            QTextEdit {
                background-color: #f8f8f8;
                border: 1px solid #ddd;
            }
        """)
        log_layout.addWidget(self.log_text)

        # Clear log button
        clear_btn = QPushButton("Clear Log")
        clear_btn.clicked.connect(self.clear_log)
        clear_btn.setMaximumWidth(100)
        log_layout.addWidget(clear_btn)

        layout.addWidget(log_group)

        # Current status at bottom
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: #0066cc; font-weight: bold; padding: 5px;")
        layout.addWidget(self.status_label)

    def add_log_entry(self, entry_type: str, message: str, details: str = ""):
        """Add an entry to the processing log"""
        timestamp = datetime.now().strftime("%H:%M:%S")

        # Format the log entry with color coding
        color_map = {
            "INFO": "#0066cc",
            "SUCCESS": "#006600",
            "WARNING": "#cc6600",
            "ERROR": "#cc0000",
            "ANALYSIS": "#6600cc",
        }

        color = color_map.get(entry_type.upper(), "#333333")

        log_entry = f"<span style='color: #666;'>[{timestamp}]</span> "
        log_entry += f"<span style='color: {color}; font-weight: bold;'>[{entry_type}]</span> "
        log_entry += f"<span style='color: #333;'>{message}</span>"

        if details:
            log_entry += f"<br/><span style='color: #666; font-style: italic; margin-left: 20px;'>{details}</span>"

        self.log_entries.append(log_entry)
        self._update_log_display()

    def _update_log_display(self):
        """Update the log display with all entries"""
        html_content = "<br/>".join(self.log_entries)
        self.log_text.setHtml(html_content)

        # Scroll to bottom
        cursor = self.log_text.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.log_text.setTextCursor(cursor)

    def clear_log(self):
        """Clear all log entries"""
        self.log_entries = []
        self.log_text.clear()
        self.add_log_entry("INFO", "Log cleared")

    def update_analysis(self, analysis_data):
        """Update the analysis display and log the results"""
        analysis = analysis_data["analysis"]

        # Update UI elements
        self.reasoning_label.setText(analysis.reasoning)

        confidence = analysis.confidence
        self.confidence_label.setText(f"Confidence: {confidence:.1%}")

        if confidence >= 0.8:
            color = "green"
        elif confidence >= 0.5:
            color = "orange"
        else:
            color = "red"
        self.confidence_label.setStyleSheet(f"font-weight: bold; color: {color};")

        # Add detailed analysis to log
        self.add_log_entry("ANALYSIS", "Query analysis completed", f"Reasoning: {analysis.reasoning}")
        self.add_log_entry("ANALYSIS", f"Confidence: {confidence:.1%}")

        if analysis.needs_search:
            self.add_log_entry("ANALYSIS", f"Search actions required: {len(analysis.search_actions)}")
            for i, action in enumerate(analysis.search_actions, 1):
                self.add_log_entry(
                    "ANALYSIS",
                    f"Action {i}: {action.action_type.value}",
                    f"Query: '{action.query}' | Priority: {action.priority}",
                )
        else:
            self.add_log_entry("ANALYSIS", "No search actions needed - can answer from knowledge")

    def update_status(self, status: str):
        """Update the current status and log it"""
        self.status_label.setText(status)

        # Determine log level based on status content
        if "error" in status.lower() or "failed" in status.lower():
            log_type = "ERROR"
        elif "complete" in status.lower() or "success" in status.lower():
            log_type = "SUCCESS"
        elif "analyzing" in status.lower() or "processing" in status.lower():
            log_type = "INFO"
        else:
            log_type = "INFO"

        self.add_log_entry(log_type, status)

    def log_search_action(self, action_type: str, query: str, result: dict):
        """Log a search action and its result"""
        status = result.get("status", "unknown")

        if status == "not_implemented":
            self.add_log_entry("WARNING", f"{action_type} search not implemented", f"Query: '{query}' | Status: {status}")
        elif status == "success":
            self.add_log_entry("SUCCESS", f"{action_type} search completed", f"Query: '{query}' | Results found")
        elif status == "error":
            error_msg = result.get("error", "Unknown error")
            self.add_log_entry("ERROR", f"{action_type} search failed", f"Query: '{query}' | Error: {error_msg}")
        else:
            self.add_log_entry("INFO", f"{action_type} search executed", f"Query: '{query}' | Status: {status}")


class MarkdownDisplay(QWebEngineView):
    """Widget for displaying markdown content with LaTeX support"""

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(300)
        self.current_content = ""

        # Configure markdown processor
        self.md = markdown.Markdown(
            extensions=["extra", "codehilite", "fenced_code", "toc"],
            extension_configs={"codehilite": {"css_class": "highlight", "use_pygments": True}},
        )

        # CSS for styling
        self.css = """
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: none;
                margin: 0;
                padding: 20px;
                background-color: #fafafa;
            }
            code {
                background-color: #f4f4f4;
                padding: 2px 4px;
                border-radius: 3px;
                font-family: 'Consolas', 'Courier New', monospace;
            }
            pre {
                background-color: #f8f8f8;
                border: 1px solid #e1e1e1;
                border-radius: 4px;
                padding: 10px;
                overflow-x: auto;
            }
            blockquote {
                border-left: 4px solid #ddd;
                margin: 0;
                padding-left: 20px;
                color: #666;
            }
            table {
                border-collapse: collapse;
                width: 100%;
                margin: 10px 0;
            }
            th, td {
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }
            th {
                background-color: #f2f2f2;
            }
            .math {
                color: #0066cc;
                font-weight: bold;
            }
            h1, h2, h3, h4, h5, h6 {
                color: #2c3e50;
                border-bottom: 1px solid #eee;
                padding-bottom: 5px;
            }
        </style>
        """

    def update_content(self, markdown_text):
        """Update the display with new markdown content"""
        self.current_content = markdown_text
        self._render()

    def append_content(self, markdown_text):
        """Append new content (for streaming)"""
        self.current_content += markdown_text
        self._render()

    def clear_content(self):
        """Clear all content"""
        self.current_content = ""
        self.setHtml("")

    def _render(self):
        """Render markdown to HTML"""
        if not self.current_content.strip():
            self.setHtml("")
            return

        content = self._process_math(self.current_content)
        html_content = self.md.convert(content)

        full_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            {self.css}
            <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
            <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
            <script>
                window.MathJax = {{
                    tex: {{
                        inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
                        displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']]
                    }}
                }};
            </script>
        </head>
        <body>
            {html_content}
        </body>
        </html>
        """
        self.setHtml(full_html)

    def _process_math(self, text):
        """Basic math processing"""
        text = re.sub(r"\$\$(.*?)\$\$", r'<div class="math">\\[\1\\]</div>', text, flags=re.DOTALL)
        text = re.sub(r"\$(.*?)\$", r'<span class="math">\\(\1\\)</span>', text)
        return text


class ConversationListWidget(QListWidget):
    """Widget for displaying conversation history"""

    def __init__(self, config_manager):
        super().__init__()
        self.config_manager = config_manager
        self.refresh_conversations()

    def refresh_conversations(self):
        """Refresh the conversation list"""
        self.clear()
        conversations = self.config_manager.load_conversations()

        for conv in conversations[:50]:  # Show latest 50 conversations
            timestamp = datetime.fromisoformat(conv["timestamp"])
            preview = conv["messages"][0]["content"][:50] + "..." if conv["messages"] else "Empty conversation"

            item_text = f"{timestamp.strftime('%Y-%m-%d %H:%M')} - {preview}"
            item = QListWidgetItem(item_text)
            item.setData(Qt.UserRole, conv)
            self.addItem(item)


class OllamaRAGGuiTabbed(QMainWindow):
    """Enhanced GUI with tabbed interface and reasoning support"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cortex - Ollama RAG Agent")

        # Configuration and components
        self.config_manager = ConfigManager()
        self.load_settings()

        # Initialize reasoning components
        self.reasoning_manager = ReasoningModelManager(self.config_manager)
        self.reasoning_parser = ReasoningParser(self.reasoning_manager)

        # Status tracking
        self.ollama_running = False
        self.available_models = []
        self.current_conversation = []

        # Status checker
        self.status_checker = OllamaStatusChecker()
        self.status_checker.status_changed.connect(self.on_ollama_status_changed)
        self.status_checker.models_updated.connect(self.on_models_updated)
        self.status_checker.connection_lost.connect(self.on_connection_lost)

        self.current_worker = None
        self.setup_ui()

        # Start status monitoring
        self.status_checker.start()

    def setup_ui(self):
        """Setup the tabbed user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout with tabs
        main_layout = QVBoxLayout(central_widget)

        # Create tab widget
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)

        # Setup individual tabs
        self.setup_chat_tab()
        self.setup_log_tab()
        self.setup_placeholder_tabs()

    def setup_chat_tab(self):
        """Setup the main chat interface tab"""
        chat_widget = QWidget()
        chat_layout = QHBoxLayout(chat_widget)

        # Left sidebar (kept from original design)
        sidebar = self.create_chat_sidebar()
        chat_layout.addWidget(sidebar)

        # Main chat area
        chat_area = QWidget()
        chat_area_layout = QVBoxLayout(chat_area)

        # Input section
        input_section = self.create_input_section()
        chat_area_layout.addWidget(input_section)

        # Output section with subtabs
        output_tabs = QTabWidget()

        # Rendered conversation tab
        self.markdown_display = EnhancedMarkdownDisplay(self.reasoning_parser)
        output_tabs.addTab(self.markdown_display, "Conversation")

        # Raw text tab with XML formatting
        self.raw_output = QTextEdit()
        self.raw_output.setReadOnly(True)
        self.raw_output.setFont(QFont("Consolas", 10))
        output_tabs.addTab(self.raw_output, "Raw Text")

        chat_area_layout.addWidget(output_tabs)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        chat_area_layout.addWidget(self.progress_bar)

        chat_layout.addWidget(chat_area)

        # Add chat tab
        self.tab_widget.addTab(chat_widget, "💬 Chat")

    def create_chat_sidebar(self):
        """Create the chat sidebar with controls"""
        sidebar = QWidget()
        sidebar.setMaximumWidth(300)
        sidebar_layout = QVBoxLayout(sidebar)

        # Status section
        status_frame = QFrame()
        status_frame.setFrameStyle(QFrame.StyledPanel)
        status_frame.setMaximumHeight(40)
        status_layout = QHBoxLayout(status_frame)

        self.status_label = QLabel("Checking Ollama status...")
        self.status_label.setStyleSheet("color: orange; font-weight: bold;")
        status_layout.addWidget(QLabel("Status:"))
        status_layout.addWidget(self.status_label)
        status_layout.addStretch()

        sidebar_layout.addWidget(status_frame)

        # Configuration buttons
        model_config_btn = QPushButton("Configure Models")
        model_config_btn.clicked.connect(self.open_model_config)
        sidebar_layout.addWidget(model_config_btn)

        prompt_config_btn = QPushButton("Configure Prompts")
        prompt_config_btn.clicked.connect(self.open_prompt_config)
        sidebar_layout.addWidget(prompt_config_btn)

        # Current model display
        model_group = QGroupBox("Current Chat Model")
        model_layout = QVBoxLayout(model_group)

        self.current_model_label = QLabel("No model selected")
        self.current_model_label.setStyleSheet("font-weight: bold;")
        model_layout.addWidget(self.current_model_label)

        # Reasoning indicator
        self.reasoning_indicator = QLabel("Reasoning: Unknown")
        self.reasoning_indicator.setStyleSheet("color: #666; font-size: 12px;")
        model_layout.addWidget(self.reasoning_indicator)

        sidebar_layout.addWidget(model_group)

        # Chat controls
        self.stream_button = QPushButton("Stream: ON")
        self.stream_button.setCheckable(True)
        self.stream_button.setChecked(self.config_manager.get("ui.stream_enabled", True))
        self.stream_button.clicked.connect(self.toggle_stream)
        self.stream_button.setEnabled(False)
        sidebar_layout.addWidget(self.stream_button)

        self.iterative_button = QPushButton("Iterative Mode: ON")
        self.iterative_button.setCheckable(True)
        self.iterative_button.setChecked(self.config_manager.get("ui.iterative_mode_enabled", True))
        self.iterative_button.clicked.connect(self.toggle_iterative_mode)
        sidebar_layout.addWidget(self.iterative_button)

        # Conversation history
        conv_group = QGroupBox("Conversation History")
        conv_layout = QVBoxLayout(conv_group)

        self.conversation_list = ConversationListWidget(self.config_manager)
        self.conversation_list.itemClicked.connect(self.load_conversation)
        conv_layout.addWidget(self.conversation_list)

        new_conv_btn = QPushButton("New Conversation")
        new_conv_btn.clicked.connect(self.new_conversation)
        conv_layout.addWidget(new_conv_btn)

        sidebar_layout.addWidget(conv_group)
        sidebar_layout.addStretch()

        return sidebar

    def create_input_section(self):
        """Create the input section"""
        input_widget = QWidget()
        input_layout = QVBoxLayout(input_widget)

        input_layout.addWidget(QLabel("Input:"))

        self.input_text = QTextEdit()
        self.input_text.setMaximumHeight(150)
        self.input_text.setPlaceholderText("Enter your message here...")
        input_layout.addWidget(self.input_text)

        # Send button layout
        button_layout = QHBoxLayout()
        self.send_button = QPushButton("Send")
        self.send_button.clicked.connect(self.send_message)
        self.send_button.setEnabled(False)
        button_layout.addWidget(self.send_button)
        button_layout.addStretch()
        input_layout.addLayout(button_layout)

        return input_widget

    def setup_log_tab(self):
        """Setup the log and analysis tab"""
        self.log_display = EnhancedLogDisplay()
        self.tab_widget.addTab(self.log_display, "📊 Analysis & Log")

    def setup_placeholder_tabs(self):
        """Setup placeholder tabs for future features"""
        # Settings tab placeholder
        settings_widget = QWidget()
        settings_layout = QVBoxLayout(settings_widget)
        settings_layout.addWidget(QLabel("Settings configuration will be implemented here"))
        self.tab_widget.addTab(settings_widget, "⚙️ Settings")

        # Workspaces tab placeholder
        workspaces_widget = QWidget()
        workspaces_layout = QVBoxLayout(workspaces_widget)
        workspaces_layout.addWidget(QLabel("Workspace and codebase management will be implemented here"))
        self.tab_widget.addTab(workspaces_widget, "📁 Workspaces")

    def update_model_display(self):
        """Update the current model display with reasoning info"""
        chat_model = self.config_manager.get("ollama.models.chat", "")
        if chat_model:
            if chat_model in self.available_models:
                self.current_model_label.setText(chat_model)
                self.current_model_label.setStyleSheet("font-weight: bold; color: green;")

                # Update reasoning indicator
                if self.reasoning_manager.is_reasoning_model(chat_model):
                    format_type = self.reasoning_manager.get_reasoning_format(chat_model)
                    self.reasoning_indicator.setText(f"Reasoning: {format_type.value}")
                    self.reasoning_indicator.setStyleSheet("color: #0066cc; font-size: 12px; font-weight: bold;")
                else:
                    self.reasoning_indicator.setText("Reasoning: Not detected")
                    self.reasoning_indicator.setStyleSheet("color: #666; font-size: 12px;")
            else:
                self.current_model_label.setText(f"{chat_model} (unavailable)")
                self.current_model_label.setStyleSheet("font-weight: bold; color: red;")
                self.reasoning_indicator.setText("Reasoning: Unknown")
        else:
            self.current_model_label.setText("No model selected")
            self.current_model_label.setStyleSheet("font-weight: bold; color: orange;")
            self.reasoning_indicator.setText("Reasoning: Unknown")

    def send_message(self):
        """Enhanced send_message with reasoning and new display"""
        if not self.ollama_running:
            self.markdown_display.add_message(
                "system", "**Error:** Ollama is not running. Please start Ollama and wait for connection."
            )
            return

        message = self.input_text.toPlainText().strip()
        if not message:
            return

        chat_model = self.config_manager.get("ollama.models.chat", "")
        if not chat_model:
            self.markdown_display.add_message("system", "**Error:** No chat model configured.")
            return

        if chat_model not in self.available_models:
            self.markdown_display.add_message("system", f"**Error:** Model '{chat_model}' not available.")
            return

        # Add user message to conversation and display
        self.current_conversation.append({"role": "user", "content": message})
        self.markdown_display.add_message("user", message)

        # Update raw text display with XML format
        self._update_raw_display()

        # Clear input
        self.input_text.clear()

        # Show progress
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.send_button.setEnabled(False)

        # Choose processing mode
        if self.iterative_button.isChecked():
            # Switch to log tab to show analysis
            self.tab_widget.setCurrentIndex(1)
            self.log_display.clear_log()

            # Use iterative processing with enhanced logging
            self.current_worker = IterativeOllamaWorker(
                self.config_manager,
                message,
                self.current_conversation[:-1],  # Exclude current message
            )

            # Connect signals
            self.current_worker.status_updated.connect(self.log_display.update_status)
            self.current_worker.analysis_completed.connect(self.log_display.update_analysis)
            self.current_worker.log_entry.connect(self.log_display.add_log_entry)

        else:
            # Use original processing with reasoning enhancement
            stream = self.stream_button.isChecked()
            system_prompt = self.config_manager.get("prompts.chat", "")

            # Add reasoning prompt if needed
            if self.reasoning_manager.is_reasoning_model(chat_model):
                reasoning_suffix = self.reasoning_manager.get_reasoning_prompt_suffix(chat_model)
                if reasoning_suffix:
                    system_prompt += reasoning_suffix

            self.current_worker = OllamaWorker(chat_model, message, stream, system_prompt)
            if hasattr(self.current_worker, "stream_chunk"):
                self.current_worker.stream_chunk.connect(self.on_stream_chunk)

        # Connect common signals
        self.current_worker.response_received.connect(self.on_response_received)
        self.current_worker.error_occurred.connect(self.on_error)
        self.current_worker.start()

    def on_stream_chunk(self, chunk):
        """Handle streaming chunk for non-iterative mode"""
        # Update the last message in markdown display
        if self.current_conversation and self.current_conversation[-1]["role"] == "assistant":
            self.current_conversation[-1]["content"] += chunk
        else:
            # Start new assistant message
            self.current_conversation.append({"role": "assistant", "content": chunk})

        # Update displays
        self.markdown_display.set_conversation(self.current_conversation)
        self._update_raw_display()

    def on_response_received(self, response):
        """Enhanced response handler with reasoning support"""
        chat_model = self.config_manager.get("ollama.models.chat", "")

        # Add or update assistant message in conversation
        if self.current_conversation and self.current_conversation[-1]["role"] == "assistant":
            # Update existing message (streaming case)
            self.current_conversation[-1]["content"] = response
        else:
            # Add new assistant message
            self.current_conversation.append({"role": "assistant", "content": response, "model_name": chat_model})

        # Learn from response if it's a reasoning model
        if self.reasoning_manager.is_reasoning_model(chat_model):
            self.reasoning_manager.learn_from_response(chat_model, response)

        # Update displays
        self.markdown_display.set_conversation(self.current_conversation)
        self._update_raw_display()

        # Auto-save conversation
        if self.config_manager.get("ui.auto_save_conversations", True):
            self.config_manager.save_conversation(self.current_conversation, chat_model)
            self.conversation_list.refresh_conversations()

        # Re-enable UI
        self.progress_bar.setVisible(False)
        self.send_button.setEnabled(True)

        # Switch back to chat tab
        self.tab_widget.setCurrentIndex(0)

    def on_error(self, error):
        """Enhanced error handler"""
        self.markdown_display.add_message("system", f"**Error:** {error}")

        # Log error if on log tab
        if self.tab_widget.currentIndex() == 1:
            self.log_display.add_log_entry("ERROR", "Processing failed", error)

        # Force model refresh on connection errors
        if "Connection" in error or "HTTP" in error:
            self.status_checker.force_model_refresh()

        # Re-enable UI
        self.progress_bar.setVisible(False)
        self.send_button.setEnabled(True)

    def _update_raw_display(self):
        """Update raw text display with XML-formatted conversation"""
        xml_parts = ["<conversation>"]

        for msg in self.current_conversation:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            timestamp = (
                msg.get("timestamp", datetime.now()).isoformat()
                if isinstance(msg.get("timestamp"), datetime)
                else datetime.now().isoformat()
            )
            model_name = msg.get("model_name", "")

            xml_parts.append(f'  <message role="{role}" timestamp="{timestamp}">')

            if model_name:
                xml_parts.append(f"    <model>{html.escape(model_name)}</model>")

            # Parse reasoning if present
            if role == "assistant" and model_name:
                reasoning_text, final_answer = self.reasoning_parser.parse_response(model_name, content)

                if reasoning_text:
                    xml_parts.append("    <thinking>")
                    xml_parts.append(f"      {html.escape(reasoning_text)}")
                    xml_parts.append("    </thinking>")
                    content = final_answer

            xml_parts.append("    <content>")
            xml_parts.append(f"      {html.escape(content)}")
            xml_parts.append("    </content>")
            xml_parts.append("  </message>")

        xml_parts.append("</conversation>")

        self.raw_output.setPlainText("\n".join(xml_parts))

    def new_conversation(self):
        """Start a new conversation"""
        if self.current_conversation and self.config_manager.get("ui.auto_save_conversations", True):
            chat_model = self.config_manager.get("ollama.models.chat", "unknown")
            self.config_manager.save_conversation(self.current_conversation, chat_model)

        self.current_conversation = []
        self.markdown_display.clear_conversation()
        self.raw_output.clear()
        self.conversation_list.refresh_conversations()

        # Switch to chat tab
        self.tab_widget.setCurrentIndex(0)

    def load_conversation(self, item):
        """Load a conversation from history"""
        conv_data = item.data(Qt.UserRole)
        if conv_data and "messages" in conv_data:
            self.current_conversation = conv_data["messages"].copy()
            self.markdown_display.set_conversation(self.current_conversation)
            self._update_raw_display()

            # Switch to chat tab
            self.tab_widget.setCurrentIndex(0)

    def toggle_stream(self):
        """Toggle streaming mode"""
        is_streaming = self.stream_button.isChecked()
        self.stream_button.setText(f"Stream: {'ON' if is_streaming else 'OFF'}")

    def toggle_iterative_mode(self):
        """Toggle iterative processing mode"""
        is_iterative = self.iterative_button.isChecked()
        self.iterative_button.setText(f"Iterative Mode: {'ON' if is_iterative else 'OFF'}")
        self.config_manager.set("ui.iterative_mode_enabled", is_iterative)

    # Status handling methods (same as original)
    def on_ollama_status_changed(self, is_running, status_message):
        """Handle Ollama status changes"""
        self.ollama_running = is_running
        self.status_label.setText(status_message)

        if is_running:
            self.status_label.setStyleSheet("color: green; font-weight: bold;")
            chat_model = self.config_manager.get("ollama.models.chat", "")
            self.send_button.setEnabled(bool(chat_model and chat_model in self.available_models))
            self.stream_button.setEnabled(True)
        else:
            self.status_label.setStyleSheet("color: red; font-weight: bold;")
            self.send_button.setEnabled(False)
            self.stream_button.setEnabled(False)

    def on_models_updated(self, models):
        """Handle updated model list from Ollama"""
        self.available_models = models
        self.update_model_display()

        # Enable send button if chat model is available
        chat_model = self.config_manager.get("ollama.models.chat", "")
        if chat_model and chat_model in models and self.ollama_running:
            self.send_button.setEnabled(True)

    def on_connection_lost(self):
        """Handle connection loss - restart polling"""
        self.status_checker.force_model_refresh()

    def open_model_config(self):
        """Open model configuration dialog"""
        if not self.available_models:
            QMessageBox.warning(
                self,
                "No Models",
                "No Ollama models are available. Please ensure Ollama is running and has models installed.",
            )
            return

        dialog = ModelConfigDialog(self.config_manager, self.available_models, self)
        if dialog.exec() == QDialog.Accepted:
            selected_models = dialog.get_selected_models()
            for task, model in selected_models.items():
                self.config_manager.set(f"ollama.models.{task}", model)
            self.config_manager.save_config()
            self.update_model_display()

    def open_prompt_config(self):
        """Open prompt configuration dialog"""
        dialog = PromptConfigDialog(self.config_manager, self)
        if dialog.exec() == QDialog.Accepted:
            prompts = dialog.get_prompts()
            for prompt_key, prompt_text in prompts.items():
                self.config_manager.set(f"prompts.{prompt_key}", prompt_text)
            self.config_manager.save_config()

    # Settings and cleanup methods (same as original)
    def load_settings(self):
        """Load application settings"""
        width = self.config_manager.get("window.width", 1400)
        height = self.config_manager.get("window.height", 900)
        x = self.config_manager.get("window.x", 100)
        y = self.config_manager.get("window.y", 100)
        self.setGeometry(x, y, width, height)

        self.ollama_url = self.config_manager.get("ollama.url", "http://localhost:11434")

    def save_settings(self):
        """Save application settings"""
        geometry = self.geometry()
        self.config_manager.set("window.width", geometry.width())
        self.config_manager.set("window.height", geometry.height())
        self.config_manager.set("window.x", geometry.x())
        self.config_manager.set("window.y", geometry.y())

        if hasattr(self, "stream_button"):
            self.config_manager.set("ui.stream_enabled", self.stream_button.isChecked())

        self.config_manager.save_config()

    def closeEvent(self, event):
        """Handle application close event"""
        # Save current conversation if it exists
        if self.current_conversation and self.config_manager.get("ui.auto_save_conversations", True):
            chat_model = self.config_manager.get("ollama.models.chat", "unknown")
            self.config_manager.save_conversation(self.current_conversation, chat_model)

        # Stop status checker
        if hasattr(self, "status_checker"):
            self.status_checker.stop()
            self.status_checker.wait(1000)

        # Save settings
        self.save_settings()

        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    window = OllamaRAGGuiTabbed()
    window.show()

    sys.exit(app.exec())

if __name__ == "__main__":
    main()
