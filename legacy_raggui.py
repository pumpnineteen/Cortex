import sys
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QFrame,
    QLabel,
    QPushButton,
    QGroupBox,
    QSplitter,
    QTextEdit,
    QTabWidget,
    QProgressBar,
    QMessageBox,
    QDialog,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QTextCursor
from config_manager import ConfigManager

class OllamaRAGGui(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cortex - Ollama RAG Agent")

        # Configuration management
        self.config_manager = ConfigManager()
        self.load_settings()

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

    def load_settings(self):
        """Load application settings"""
        width = self.config_manager.get("window.width", 1400)
        height = self.config_manager.get("window.height", 900)
        x = self.config_manager.get("window.x", 100)
        y = self.config_manager.get("window.y", 100)
        self.setGeometry(x, y, width, height)

        self.ollama_url = self.config_manager.get("ollama.url", "http://localhost:11434")
        self.stream_enabled = self.config_manager.get("ui.stream_enabled", True)

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

    def setup_ui(self):
        """Setup the user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main horizontal layout
        main_layout = QHBoxLayout(central_widget)

        # Left sidebar for conversations and controls
        left_panel = QWidget()
        left_panel.setMaximumWidth(300)
        left_layout = QVBoxLayout(left_panel)

        # Status bar
        status_frame = QFrame()
        status_frame.setFrameStyle(QFrame.StyledPanel)
        status_frame.setMaximumHeight(40)
        status_layout = QHBoxLayout(status_frame)

        self.status_label = QLabel("Checking Ollama status...")
        self.status_label.setStyleSheet("color: orange; font-weight: bold;")
        status_layout.addWidget(QLabel("Status:"))
        status_layout.addWidget(self.status_label)
        status_layout.addStretch()

        left_layout.addWidget(status_frame)

        # Model configuration button
        model_config_btn = QPushButton("Configure Models")
        model_config_btn.clicked.connect(self.open_model_config)
        left_layout.addWidget(model_config_btn)

        # Prompt configuration button
        prompt_config_btn = QPushButton("Configure Prompts")
        prompt_config_btn.clicked.connect(self.open_prompt_config)
        left_layout.addWidget(prompt_config_btn)

        # Current model display
        current_model_group = QGroupBox("Current Chat Model")
        current_model_layout = QVBoxLayout(current_model_group)
        self.current_model_label = QLabel("No model selected")
        self.current_model_label.setStyleSheet("font-weight: bold;")
        current_model_layout.addWidget(self.current_model_label)
        left_layout.addWidget(current_model_group)

        # Stream toggle
        self.stream_button = QPushButton("Stream: ON")
        self.stream_button.setCheckable(True)
        self.stream_button.setChecked(self.stream_enabled)
        self.stream_button.clicked.connect(self.toggle_stream)
        self.stream_button.setEnabled(False)
        self.stream_button.setText(f"Stream: {'ON' if self.stream_enabled else 'OFF'}")
        left_layout.addWidget(self.stream_button)

        # Conversation history
        conv_group = QGroupBox("Conversation History")
        conv_layout = QVBoxLayout(conv_group)

        self.conversation_list = ConversationListWidget(self.config_manager)
        self.conversation_list.itemClicked.connect(self.load_conversation)
        conv_layout.addWidget(self.conversation_list)

        new_conv_btn = QPushButton("New Conversation")
        new_conv_btn.clicked.connect(self.new_conversation)
        conv_layout.addWidget(new_conv_btn)

        left_layout.addWidget(conv_group)
        left_layout.addStretch()

        main_layout.addWidget(left_panel)

        # Right panel for chat interface
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # Main content area with splitter
        splitter = QSplitter(Qt.Vertical)

        # Input area
        input_widget = QWidget()
        input_layout = QVBoxLayout(input_widget)
        input_layout.addWidget(QLabel("Input:"))

        self.input_text = QTextEdit()
        self.input_text.setMaximumHeight(150)
        self.input_text.setPlaceholderText("Enter your message here...")
        input_layout.addWidget(self.input_text)

        # Send button
        button_layout = QHBoxLayout()
        self.send_button = QPushButton("Send")
        self.send_button.clicked.connect(self.send_message)
        self.send_button.setEnabled(False)
        button_layout.addWidget(self.send_button)
        button_layout.addStretch()
        input_layout.addLayout(button_layout)

        splitter.addWidget(input_widget)

        # Output area with tabs
        self.output_tabs = QTabWidget()

        # Rendered markdown tab
        self.markdown_display = MarkdownDisplay()
        self.output_tabs.addTab(self.markdown_display, "Rendered")

        # Raw text tab
        self.raw_output = QTextEdit()
        self.raw_output.setReadOnly(True)
        self.raw_output.setFont(QFont("Consolas", 10))
        self.output_tabs.addTab(self.raw_output, "Raw Text")

        # JSON tab (for debugging)
        self.json_output = QTextEdit()
        self.json_output.setReadOnly(True)
        self.json_output.setFont(QFont("Consolas", 10))
        self.output_tabs.addTab(self.json_output, "Debug")

        splitter.addWidget(self.output_tabs)
        splitter.setSizes([200, 600])

        right_layout.addWidget(splitter)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        right_layout.addWidget(self.progress_bar)

        main_layout.addWidget(right_panel)

        # Update UI state
        self.update_model_display()

    def update_model_display(self):
        """Update the current model display"""
        chat_model = self.config_manager.get("ollama.models.chat", "")
        if chat_model:
            if chat_model in self.available_models:
                self.current_model_label.setText(chat_model)
                self.current_model_label.setStyleSheet("font-weight: bold; color: green;")
            else:
                self.current_model_label.setText(f"{chat_model} (unavailable)")
                self.current_model_label.setStyleSheet("font-weight: bold; color: red;")
        else:
            self.current_model_label.setText("No model selected")
            self.current_model_label.setStyleSheet("font-weight: bold; color: orange;")

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

    def new_conversation(self):
        """Start a new conversation"""
        if self.current_conversation and self.config_manager.get("ui.auto_save_conversations", True):
            chat_model = self.config_manager.get("ollama.models.chat", "unknown")
            self.config_manager.save_conversation(self.current_conversation, chat_model)

        self.current_conversation = []
        self.markdown_display.clear_content()
        self.raw_output.clear()
        self.json_output.clear()
        self.conversation_list.refresh_conversations()

    def load_conversation(self, item):
        """Load a conversation from history"""
        conv_data = item.data(Qt.UserRole)
        if conv_data and "messages" in conv_data:
            # Display the conversation
            content = ""
            for msg in conv_data["messages"]:
                role = msg.get("role", "unknown")
                text = msg.get("content", "")
                content += f"**{role.title()}:** {text}\n\n"

            self.markdown_display.update_content(content)
            self.current_conversation = conv_data["messages"].copy()

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

    def toggle_stream(self):
        """Toggle streaming mode"""
        is_streaming = self.stream_button.isChecked()
        self.stream_button.setText(f"Stream: {'ON' if is_streaming else 'OFF'}")

    def send_message(self):
        """Send message to Ollama"""
        if not self.ollama_running:
            self.markdown_display.update_content(
                "**Error:** Ollama is not running. Please start Ollama and wait for connection."
            )
            return

        message = self.input_text.toPlainText().strip()
        if not message:
            return

        chat_model = self.config_manager.get("ollama.models.chat", "")
        if not chat_model:
            self.markdown_display.update_content("**Error:** No chat model configured. Please configure models first.")
            return

        if chat_model not in self.available_models:
            self.markdown_display.update_content(
                f"**Error:** Model '{chat_model}' is not available. Please check your model configuration."
            )
            return

        stream = self.stream_button.isChecked()
        system_prompt = self.config_manager.get("prompts.chat", "")

        # Add user message to conversation
        self.current_conversation.append({"role": "user", "content": message})

        # Clear previous output and show user message
        self.markdown_display.update_content(f"**You:** {message}\n\n**Assistant:** ")
        self.raw_output.clear()
        self.json_output.clear()

        # Clear input
        self.input_text.clear()

        # Show progress
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.send_button.setEnabled(False)

        # Start worker thread
        self.current_worker = OllamaWorker(chat_model, message, stream, system_prompt)
        self.current_worker.response_received.connect(self.on_response_received)
        self.current_worker.error_occurred.connect(self.on_error)
        self.current_worker.stream_chunk.connect(self.on_stream_chunk)
        self.current_worker.start()

    def on_stream_chunk(self, chunk):
        """Handle streaming chunk"""
        self.markdown_display.append_content(chunk)

        # Update raw text
        self.raw_output.moveCursor(QTextCursor.End)
        self.raw_output.insertPlainText(chunk)

    def on_response_received(self, response):
        """Handle complete response"""
        # Add assistant response to conversation
        self.current_conversation.append({"role": "assistant", "content": response})

        # Auto-save if enabled
        if self.config_manager.get("ui.auto_save_conversations", True):
            chat_model = self.config_manager.get("ollama.models.chat", "")
            self.config_manager.save_conversation(self.current_conversation, chat_model)
            self.conversation_list.refresh_conversations()

        self.json_output.setPlainText(
            f"Response length: {len(response)} characters\nModel: {self.config_manager.get('ollama.models.chat', 'unknown')}"
        )

        # Hide progress and re-enable send
        self.progress_bar.setVisible(False)
        chat_model = self.config_manager.get("ollama.models.chat", "")
        self.send_button.setEnabled(bool(chat_model and chat_model in self.available_models))

    def on_error(self, error):
        """Handle error"""
        self.markdown_display.append_content(f"\n\n**Error:** {error}")
        self.json_output.setPlainText(f"Error: {error}")

        # Force model refresh on connection errors
        if "Connection" in error or "HTTP" in error:
            self.status_checker.force_model_refresh()

        # Hide progress and re-enable send
        self.progress_bar.setVisible(False)
        chat_model = self.config_manager.get("ollama.models.chat", "")
        self.send_button.setEnabled(bool(chat_model and chat_model in self.available_models))

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


class OllamaRAGGuiExtended(OllamaRAGGui):
    """Extended GUI with iterative chat support and enhanced logging"""

    def setup_ui(self):
        """Extended UI setup with analysis display"""
        # Call the parent setup_ui first
        super().setup_ui()

        # Add the analysis display to the right panel
        right_panel = self.centralWidget().layout().itemAt(1).widget()
        right_layout = right_panel.layout()

        # Insert analysis widget before the splitter
        self.analysis_display = AnalysisDisplayWidget()
        right_layout.insertWidget(0, self.analysis_display)

        # Add toggle for iterative mode to left panel
        left_panel = self.centralWidget().layout().itemAt(0).widget()
        left_layout = left_panel.layout()

        # Insert after the stream button
        self.iterative_button = QPushButton("Iterative Mode: ON")
        self.iterative_button.setCheckable(True)
        self.iterative_button.setChecked(self.config_manager.get("ui.iterative_mode_enabled", True))
        self.iterative_button.clicked.connect(self.toggle_iterative_mode)

        # Find stream button and insert iterative button after it
        for i in range(left_layout.count()):
            widget = left_layout.itemAt(i).widget()
            if hasattr(widget, "text") and widget.text() and "Stream:" in widget.text():
                left_layout.insertWidget(i + 1, self.iterative_button)
                break

        # Set initial visibility based on iterative mode
        is_iterative = self.iterative_button.isChecked()
        self.analysis_display.setVisible(is_iterative)
        self.iterative_button.setText(f"Iterative Mode: {'ON' if is_iterative else 'OFF'}")

    def toggle_iterative_mode(self):
        """Toggle iterative processing mode"""
        is_iterative = self.iterative_button.isChecked()
        self.iterative_button.setText(f"Iterative Mode: {'ON' if is_iterative else 'OFF'}")
        self.analysis_display.setVisible(is_iterative)

        # Save setting
        self.config_manager.set("ui.iterative_mode_enabled", is_iterative)

    def send_message(self):
        """Enhanced send_message with iterative processing"""
        if not self.ollama_running:
            self.markdown_display.update_content(
                "**Error:** Ollama is not running. Please start Ollama and wait for connection."
            )
            return

        message = self.input_text.toPlainText().strip()
        if not message:
            return

        chat_model = self.config_manager.get("ollama.models.chat", "")
        if not chat_model:
            self.markdown_display.update_content("**Error:** No chat model configured.")
            return

        if chat_model not in self.available_models:
            self.markdown_display.update_content(f"**Error:** Model '{chat_model}' not available.")
            return

        # Add user message to conversation
        self.current_conversation.append({"role": "user", "content": message})

        # Clear previous output and show user message
        self.markdown_display.update_content(f"**You:** {message}\n\n")
        self.raw_output.clear()
        self.json_output.clear()
        self.input_text.clear()

        # Show progress
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.send_button.setEnabled(False)

        # Choose processing mode
        if self.iterative_button.isChecked():
            # Clear previous log and start fresh
            self.analysis_display.clear_log()

            # Use iterative processing with log callback
            self.current_worker = IterativeOllamaWorker(
                self.config_manager,
                message,
                self.current_conversation[:-1],  # Exclude current message
            )

            # Connect enhanced logging signals
            self.current_worker.status_updated.connect(self.analysis_display.update_status)
            self.current_worker.analysis_completed.connect(self.analysis_display.update_analysis)
            self.current_worker.log_entry.connect(self.analysis_display.add_log_entry)

        else:
            # Use original processing
            stream = self.stream_button.isChecked()
            system_prompt = self.config_manager.get("prompts.chat", "")

            self.current_worker = OllamaWorker(chat_model, message, stream, system_prompt)
            if hasattr(self.current_worker, "stream_chunk"):
                self.current_worker.stream_chunk.connect(self.on_stream_chunk)

        # Connect common signals
        self.current_worker.response_received.connect(self.on_response_received)
        self.current_worker.error_occurred.connect(self.on_error)
        self.current_worker.start()

    def on_response_received(self, response):
        """Enhanced response handler"""
        # Update markdown display
        current_content = self.markdown_display.current_content
        if not current_content.endswith("**Assistant:** "):
            self.markdown_display.append_content("**Assistant:** ")

        self.markdown_display.append_content(response)
        self.raw_output.setPlainText(response)

        # Add to conversation
        self.current_conversation.append({"role": "assistant", "content": response})

        # Auto-save if enabled
        if self.config_manager.get("ui.auto_save_conversations", True):
            chat_model = self.config_manager.get("ollama.models.chat", "")
            self.config_manager.save_conversation(self.current_conversation, chat_model)
            self.conversation_list.refresh_conversations()

        # Update debug info
        mode = "Iterative" if self.iterative_button.isChecked() else "Direct"
        self.json_output.setPlainText(
            f"Mode: {mode}\n"
            f"Model: {self.config_manager.get('ollama.models.chat', 'unknown')}\n"
            f"Response length: {len(response)} chars\n"
            f"Conversation length: {len(self.current_conversation)} messages"
        )

        # Re-enable UI
        self.progress_bar.setVisible(False)
        self.send_button.setEnabled(True)

    def on_error(self, error):
        """Enhanced error handler"""
        self.markdown_display.append_content(f"\n\n**Error:** {error}")

        # Log error if in iterative mode
        if hasattr(self, "analysis_display") and self.iterative_button.isChecked():
            self.analysis_display.add_log_entry("ERROR", "Processing failed", error)

        # Re-enable UI
        self.progress_bar.setVisible(False)
        self.send_button.setEnabled(True)


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    window = OllamaRAGGuiExtended()
    window.show()

    sys.exit(app.exec())
