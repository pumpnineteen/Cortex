import re
import html
import markdown
from markdown.extensions import codehilite, fenced_code, tables, toc
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtCore import QUrl
from typing import List, Dict, Tuple, Optional
from datetime import datetime


class EnhancedMarkdownDisplay(QWebEngineView):
    """Enhanced widget for displaying markdown content with dialogue support and reasoning chains"""

    def __init__(self, reasoning_parser=None):
        super().__init__()
        self.setMinimumHeight(300)
        self.current_conversation = []
        self.reasoning_parser = reasoning_parser

        # Configure markdown processor with more extensions
        self.md = markdown.Markdown(
            extensions=[
                "extra",
                "codehilite",
                "fenced_code",
                "tables",
                "toc",
                "nl2br",  # Convert newlines to <br>
                "sane_lists",  # Better list handling
            ],
            extension_configs={
                "codehilite": {"css_class": "highlight", "use_pygments": True, "guess_lang": True},
                "toc": {"permalink": True},
            },
        )

        # Enhanced CSS with dialogue styling
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
            
            /* Message containers */
            .message {
                margin: 20px 0;
                padding: 15px;
                border-radius: 8px;
                border-left: 4px solid;
            }
            
            .message.user {
                background-color: #e3f2fd;
                border-left-color: #2196f3;
            }
            
            .message.assistant {
                background-color: #f1f8e9;
                border-left-color: #4caf50;
            }
            
            /* Role headers */
            .role-header {
                font-weight: bold;
                font-size: 1.1em;
                margin-bottom: 10px;
                display: flex;
                align-items: center;
                gap: 8px;
            }
            
            .role-header.user {
                color: #1976d2;
            }
            
            .role-header.assistant {
                color: #388e3c;
            }
            
            /* Reasoning sections */
            .reasoning-section {
                background-color: #fff3e0;
                border: 1px solid #ffcc02;
                border-radius: 6px;
                margin: 10px 0;
                overflow: hidden;
            }
            
            .reasoning-header {
                background-color: #ffcc02;
                color: #d84315;
                font-weight: bold;
                padding: 8px 12px;
                cursor: pointer;
                user-select: none;
            }
            
            .reasoning-content {
                padding: 12px;
                font-style: italic;
                color: #5d4037;
                white-space: pre-wrap;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 0.9em;
            }
            
            /* Code styling */
            code {
                background-color: #f4f4f4;
                padding: 2px 6px;
                border-radius: 3px;
                font-family: 'Consolas', 'Courier New', monospace;
                color: #d73a49;
            }
            
            pre {
                background-color: #f8f8f8;
                border: 1px solid #e1e1e1;
                border-radius: 6px;
                padding: 15px;
                overflow-x: auto;
                line-height: 1.4;
            }
            
            pre code {
                background: none;
                padding: 0;
                color: inherit;
            }
            
            /* Blockquotes */
            blockquote {
                border-left: 4px solid #ddd;
                margin: 15px 0;
                padding: 0 20px;
                color: #666;
                font-style: italic;
            }
            
            /* Tables */
            table {
                border-collapse: collapse;
                width: 100%;
                margin: 15px 0;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            }
            
            th, td {
                border: 1px solid #ddd;
                padding: 10px;
                text-align: left;
            }
            
            th {
                background-color: #f2f2f2;
                font-weight: bold;
            }
            
            /* Headers */
            h1, h2, h3, h4, h5, h6 {
                color: #2c3e50;
                margin-top: 1.5em;
                margin-bottom: 0.5em;
                line-height: 1.2;
            }
            
            h1 { border-bottom: 2px solid #3498db; padding-bottom: 10px; }
            h2 { border-bottom: 1px solid #bdc3c7; padding-bottom: 5px; }
            
            /* Lists */
            ul, ol {
                padding-left: 25px;
                margin: 10px 0;
            }
            
            li {
                margin: 5px 0;
            }
            
            /* Links */
            a {
                color: #3498db;
                text-decoration: none;
            }
            
            a:hover {
                text-decoration: underline;
            }
            
            /* Timestamp */
            .timestamp {
                font-size: 0.8em;
                color: #999;
                float: right;
                font-weight: normal;
            }
            
            /* Separator */
            .message-separator {
                height: 1px;
                background: linear-gradient(to right, transparent, #ddd, transparent);
                margin: 30px 0;
                border: none;
            }
            
            /* Math support */
            .math {
                color: #0066cc;
                font-weight: bold;
            }
            
            /* Collapsible reasoning */
            .reasoning-section.collapsed .reasoning-content {
                display: none;
            }
            
            .reasoning-section:not(.collapsed) .reasoning-header::after {
                content: ' ▼';
                float: right;
            }
            
            .reasoning-section.collapsed .reasoning-header::after {
                content: ' ▶';
                float: right;
            }
        </style>
        
        <script>
            function toggleReasoning(element) {
                element.classList.toggle('collapsed');
            }
            
            document.addEventListener('DOMContentLoaded', function() {
                // Make reasoning sections collapsible
                document.querySelectorAll('.reasoning-header').forEach(header => {
                    header.addEventListener('click', function() {
                        toggleReasoning(this.parentElement);
                    });
                });
            });
        </script>
        """

    def parse_user_content(self, content: str) -> str:
        """Parse and enhance user content, converting code snippets to proper markdown"""
        # Look for inline code patterns and enhance them
        enhanced_content = content

        # Convert common patterns to proper markdown
        # File paths
        enhanced_content = re.sub(r'([a-zA-Z]:\\[^\\/:*?"<>|\s]+(?:\\[^\\/:*?"<>|\s]*)*)', r"`\1`", enhanced_content)
        enhanced_content = re.sub(r'(/[^\s:*?"<>|]+(?:/[^\s:*?"<>|]*)*)', r"`\1`", enhanced_content)

        # Function calls and method names
        enhanced_content = re.sub(r"\b([a-zA-Z_][a-zA-Z0-9_]*\([^)]*\))", r"`\1`", enhanced_content)

        # Variable names and identifiers (be careful not to over-match)
        enhanced_content = re.sub(r"\b([a-zA-Z_][a-zA-Z0-9_]*\.[a-zA-Z_][a-zA-Z0-9_]*)\b", r"`\1`", enhanced_content)

        # URLs
        enhanced_content = re.sub(r'(https?://[^\s<>"]+)', r"[\1](\1)", enhanced_content)

        return enhanced_content

    def add_message(self, role: str, content: str, timestamp: Optional[datetime] = None, model_name: str = ""):
        """Add a message to the conversation"""
        if timestamp is None:
            timestamp = datetime.now()

        message = {"role": role, "content": content, "timestamp": timestamp, "model_name": model_name}

        self.current_conversation.append(message)
        self._update_display()

    def set_conversation(self, conversation: List[Dict]):
        """Set the entire conversation"""
        self.current_conversation = conversation.copy()
        self._update_display()

    def clear_conversation(self):
        """Clear all messages"""
        self.current_conversation = []
        self.setHtml("")

    def _update_display(self):
        """Update the display with current conversation"""
        if not self.current_conversation:
            self.setHtml("")
            return

        html_content = self._build_conversation_html()
        self.setHtml(html_content)

        # Scroll to bottom
        self.page().runJavaScript("window.scrollTo(0, document.body.scrollHeight);")

    def _build_conversation_html(self) -> str:
        """Build HTML for the entire conversation"""
        messages_html = []

        for i, message in enumerate(self.current_conversation):
            role = message.get("role", "unknown")
            content = message.get("content", "")
            timestamp = message.get("timestamp", datetime.now())
            model_name = message.get("model_name", "")

            # Parse reasoning if this is an assistant message
            reasoning_text = None
            final_answer = content

            if role == "assistant" and self.reasoning_parser and model_name:
                reasoning_text, final_answer = self.reasoning_parser.parse_response(model_name, content)

            # Process content based on role
            if role == "user":
                processed_content = self.parse_user_content(final_answer)
            else:
                processed_content = final_answer

            # Convert to markdown
            markdown_content = self.md.convert(processed_content)

            # Build message HTML
            message_html = self._build_message_html(role, markdown_content, timestamp, reasoning_text, model_name)
            messages_html.append(message_html)

            # Add separator except for last message
            if i < len(self.current_conversation) - 1:
                messages_html.append('<hr class="message-separator">')

        full_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            {self.css}
        </head>
        <body>
            {"".join(messages_html)}
        </body>
        </html>
        """

        return full_html

    def _build_message_html(
        self, role: str, content: str, timestamp: datetime, reasoning_text: Optional[str] = None, model_name: str = ""
    ) -> str:
        """Build HTML for a single message"""

        # Role emoji and display name
        role_info = {
            "user": ("👤", "You"),
            "assistant": ("🤖", f"Assistant{f' ({model_name})' if model_name else ''}"),
            "system": ("⚙️", "System"),
        }

        emoji, display_name = role_info.get(role, ("❓", role.title()))

        timestamp_str = timestamp.strftime("%H:%M:%S")

        message_html = f"""
        <div class="message {role}">
            <div class="role-header {role}">
                <span class="emoji">{emoji}</span>
                <span class="name">{display_name}</span>
                <span class="timestamp">{timestamp_str}</span>
            </div>
            <div class="content">
        """

        # Add reasoning section if present
        if reasoning_text:
            escaped_reasoning = html.escape(reasoning_text)
            message_html += f"""
                <div class="reasoning-section collapsed">
                    <div class="reasoning-header" onclick="toggleReasoning(this.parentElement)">
                        💭 Chain of Thought
                    </div>
                    <div class="reasoning-content">{escaped_reasoning}</div>
                </div>
            """

        # Add main content
        message_html += f"""
                {content}
            </div>
        </div>
        """

        return message_html

    def append_to_last_message(self, content: str):
        """Append content to the last message (for streaming)"""
        if not self.current_conversation:
            return

        self.current_conversation[-1]["content"] += content
        self._update_display()

    def update_last_message(self, content: str):
        """Update the content of the last message"""
        if not self.current_conversation:
            return

        self.current_conversation[-1]["content"] = content
        self._update_display()

    def export_conversation(self, format: str = "markdown") -> str:
        """Export conversation in various formats"""
        if format == "markdown":
            return self._export_markdown()
        elif format == "html":
            return self._build_conversation_html()
        elif format == "text":
            return self._export_plain_text()
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def _export_markdown(self) -> str:
        """Export as markdown"""
        markdown_parts = []

        for message in self.current_conversation:
            role = message.get("role", "unknown")
            content = message.get("content", "")
            timestamp = message.get("timestamp", datetime.now())
            model_name = message.get("model_name", "")

            # Parse reasoning
            reasoning_text = None
            final_answer = content

            if role == "assistant" and self.reasoning_parser and model_name:
                reasoning_text, final_answer = self.reasoning_parser.parse_response(model_name, content)

            # Role header
            role_emoji = {"user": "👤", "assistant": "🤖", "system": "⚙️"}.get(role, "❓")
            role_name = {"user": "You", "assistant": "Assistant", "system": "System"}.get(role, role.title())

            timestamp_str = timestamp.strftime("%H:%M:%S")
            markdown_parts.append(f"## {role_emoji} {role_name} *({timestamp_str})*")

            # Add reasoning if present
            if reasoning_text:
                markdown_parts.append(
                    f"\n<details>\n<summary>💭 Chain of Thought</summary>\n\n```\n{reasoning_text}\n```\n\n</details>\n"
                )

            # Add content
            if role == "user":
                processed_content = self.parse_user_content(final_answer)
            else:
                processed_content = final_answer

            markdown_parts.append(processed_content)
            markdown_parts.append("\n---\n")

        return "\n\n".join(markdown_parts)

    def _export_plain_text(self) -> str:
        """Export as plain text"""
        text_parts = []

        for message in self.current_conversation:
            role = message.get("role", "unknown")
            content = message.get("content", "")
            timestamp = message.get("timestamp", datetime.now())
            model_name = message.get("model_name", "")

            timestamp_str = timestamp.strftime("%H:%M:%S")
            role_name = {"user": "You", "assistant": "Assistant", "system": "System"}.get(role, role.title())

            text_parts.append(f"[{timestamp_str}] {role_name}: {content}")

        return "\n\n".join(text_parts)
