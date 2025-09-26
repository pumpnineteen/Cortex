import re
import html
import traceback
from datetime import datetime
from typing import List, Optional, Dict, Any
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QTextEdit,
    QPushButton,
    QLabel,
    QGroupBox,
    QScrollArea,
    QCheckBox,
    QComboBox,
    QSplitter,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QTextCursor


class EnhancedLogDisplay(QWidget):
    """Enhanced log display with proper formatting and filtering"""

    def __init__(self):
        super().__init__()
        self.log_entries = []
        self.filtered_entries = []
        self.setup_ui()

    def setup_ui(self):
        """Setup the enhanced log display UI"""
        layout = QVBoxLayout(self)

        # Controls section
        controls_group = QGroupBox("Log Controls")
        controls_layout = QHBoxLayout(controls_group)

        # Filter by log level
        self.level_filter = QComboBox()
        self.level_filter.addItems(["All", "INFO", "SUCCESS", "WARNING", "ERROR", "ANALYSIS"])
        self.level_filter.currentTextChanged.connect(self._apply_filters)
        controls_layout.addWidget(QLabel("Level:"))
        controls_layout.addWidget(self.level_filter)

        # Auto-scroll checkbox
        self.auto_scroll_cb = QCheckBox("Auto-scroll")
        self.auto_scroll_cb.setChecked(True)
        controls_layout.addWidget(self.auto_scroll_cb)

        # Clear button
        clear_btn = QPushButton("Clear Log")
        clear_btn.clicked.connect(self.clear_log)
        controls_layout.addWidget(clear_btn)

        # Export button
        export_btn = QPushButton("Export Log")
        export_btn.clicked.connect(self.export_log)
        controls_layout.addWidget(export_btn)

        controls_layout.addStretch()
        layout.addWidget(controls_group)

        # Splitter for analysis and log
        splitter = QSplitter(Qt.Horizontal)

        # Query analysis summary (left side)
        analysis_group = QGroupBox("Query Analysis")
        analysis_layout = QVBoxLayout(analysis_group)

        # Analysis info layout
        info_layout = QHBoxLayout()

        self.reasoning_label = QLabel("No analysis yet")
        self.reasoning_label.setWordWrap(True)
        self.reasoning_label.setStyleSheet("font-style: italic; color: #666; padding: 5px;")
        info_layout.addWidget(self.reasoning_label, stretch=3)

        # Analysis metadata
        metadata_widget = QWidget()
        metadata_layout = QVBoxLayout(metadata_widget)
        metadata_layout.setSpacing(5)

        self.confidence_label = QLabel("Confidence: N/A")
        self.confidence_label.setStyleSheet("font-weight: bold;")
        metadata_layout.addWidget(self.confidence_label)

        self.query_type_label = QLabel("Type: Unknown")
        self.query_type_label.setStyleSheet("color: #666;")
        metadata_layout.addWidget(self.query_type_label)

        self.complexity_label = QLabel("Complexity: Unknown")
        self.complexity_label.setStyleSheet("color: #666;")
        metadata_layout.addWidget(self.complexity_label)

        info_layout.addWidget(metadata_widget, stretch=1)
        analysis_layout.addLayout(info_layout)

        # Search actions summary
        self.search_actions_label = QLabel("No search actions")
        self.search_actions_label.setStyleSheet("font-size: 12px; color: #555; padding: 5px;")
        analysis_layout.addWidget(self.search_actions_label)

        analysis_group.setMaximumWidth(350)
        splitter.addWidget(analysis_group)

        # Detailed processing log (right side)
        log_group = QGroupBox("Processing Log")
        log_layout = QVBoxLayout(log_group)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Consolas", 9))
        self.log_text.setStyleSheet("""
            QTextEdit {
                background-color: #f8f8f8;
                border: 1px solid #ddd;
                border-radius: 4px;
            }
        """)
        log_layout.addWidget(self.log_text)

        splitter.addWidget(log_group)
        splitter.setSizes([350, 600])  # Give more space to log

        layout.addWidget(splitter)

        # Current status at bottom
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("""
            QLabel {
                color: #0066cc; 
                font-weight: bold; 
                padding: 8px;
                background-color: #f0f8ff;
                border: 1px solid #ddd;
                border-radius: 4px;
            }
        """)
        layout.addWidget(self.status_label)

    def add_log_entry(self, entry_type: str, message: str, details: str = ""):
        """Add an entry to the processing log with proper formatting"""
        timestamp = datetime.now()

        log_entry = {"timestamp": timestamp, "type": entry_type.upper(), "message": message, "details": details}

        self.log_entries.append(log_entry)
        self._apply_filters()

    def _apply_filters(self):
        """Apply current filters and update display"""
        level_filter = self.level_filter.currentText()

        if level_filter == "All":
            self.filtered_entries = self.log_entries.copy()
        else:
            self.filtered_entries = [entry for entry in self.log_entries if entry["type"] == level_filter]

        self._update_log_display()

    def _update_log_display(self):
        """Update the log display with filtered entries"""
        html_parts = []

        for entry in self.filtered_entries:
            formatted_entry = self._format_log_entry(entry)
            html_parts.append(formatted_entry)

        html_content = "".join(html_parts)
        self.log_text.setHtml(html_content)

        # Auto-scroll to bottom if enabled
        if self.auto_scroll_cb.isChecked():
            cursor = self.log_text.textCursor()
            cursor.movePosition(QTextCursor.End)
            self.log_text.setTextCursor(cursor)

    def _format_log_entry(self, entry: Dict[str, Any]) -> str:
        """Format a single log entry with proper HTML formatting"""
        timestamp = entry["timestamp"].strftime("%H:%M:%S.%f")[:-3]  # Include milliseconds
        entry_type = entry["type"]
        message = entry["message"]
        details = entry["details"]

        # Color coding for different log levels
        color_map = {
            "INFO": "#0066cc",
            "SUCCESS": "#006600",
            "WARNING": "#cc6600",
            "ERROR": "#cc0000",
            "ANALYSIS": "#6600cc",
        }

        color = color_map.get(entry_type, "#333333")

        # Format the main log line
        log_line = f"""
        <div style="margin-bottom: 8px; padding: 4px; border-left: 3px solid {color}; background-color: rgba({self._hex_to_rgb(color)}, 0.05);">
            <span style="color: #666; font-family: monospace; font-size: 11px;">[{timestamp}]</span>
            <span style="color: {color}; font-weight: bold; margin-left: 8px;">[{entry_type}]</span>
            <span style="color: #333; margin-left: 8px;">{html.escape(message)}</span>
        """

        # Handle details with special formatting for different content types
        if details:
            if entry_type == "ERROR" and "Traceback" in details:
                # Special formatting for tracebacks
                formatted_details = self._format_traceback(details)
            elif details.startswith("{") and details.endswith("}"):
                # JSON-like content
                formatted_details = self._format_json_like(details)
            elif "\n" in details:
                # Multi-line content
                formatted_details = self._format_multiline(details)
            else:
                # Simple details
                formatted_details = f'<span style="color: #666; font-style: italic;">{html.escape(details)}</span>'

            log_line += f"""
            <div style="margin-top: 4px; margin-left: 20px; padding: 8px; background-color: rgba(0,0,0,0.02); border-radius: 4px; font-family: monospace; font-size: 12px;">
                {formatted_details}
            </div>
            """

        log_line += "</div>"
        return log_line

    def _format_traceback(self, traceback_text: str) -> str:
        """Format traceback with syntax highlighting"""
        lines = traceback_text.split("\n")
        formatted_lines = []

        for line in lines:
            escaped_line = html.escape(line)

            if line.strip().startswith("Traceback"):
                # Header
                formatted_lines.append(f'<div style="color: #cc0000; font-weight: bold;">{escaped_line}</div>')
            elif line.strip().startswith("File "):
                # File references
                formatted_lines.append(f'<div style="color: #0066cc; font-weight: bold;">{escaped_line}</div>')
            elif any(keyword in line for keyword in ["Error:", "Exception:", "raise "]):
                # Error lines
                formatted_lines.append(f'<div style="color: #cc0000; font-weight: bold;">{escaped_line}</div>')
            elif line.strip().startswith(">>>") or line.strip().startswith("..."):
                # Python prompt
                formatted_lines.append(f'<div style="color: #666;">{escaped_line}</div>')
            else:
                # Regular lines
                formatted_lines.append(f"<div>{escaped_line}</div>")

        return "".join(formatted_lines)

    def _format_json_like(self, content: str) -> str:
        """Format JSON-like content with basic highlighting"""
        try:
            import json

            # Try to parse and pretty-print JSON
            parsed = json.loads(content)
            formatted = json.dumps(parsed, indent=2)
            escaped = html.escape(formatted)

            # Basic syntax highlighting
            escaped = re.sub(r'"([^"]+)":', r'<span style="color: #0066cc;">"\1"</span>:', escaped)
            escaped = re.sub(r': "([^"]*)"', r': <span style="color: #008000;">"\1"</span>', escaped)
            escaped = re.sub(r": (\d+)", r': <span style="color: #cc6600;">\1</span>', escaped)
            escaped = re.sub(r": (true|false|null)", r': <span style="color: #800080;">\1</span>', escaped)

            return f'<pre style="margin: 0; white-space: pre-wrap;">{escaped}</pre>'
        except:
            # If not valid JSON, treat as regular text
            return f'<pre style="margin: 0; white-space: pre-wrap; color: #666;">{html.escape(content)}</pre>'

    def _format_multiline(self, content: str) -> str:
        """Format multi-line content preserving structure"""
        escaped = html.escape(content)
        return f'<pre style="margin: 0; white-space: pre-wrap; color: #555;">{escaped}</pre>'

    def _hex_to_rgb(self, hex_color: str) -> str:
        """Convert hex color to RGB values for alpha backgrounds"""
        hex_color = hex_color.lstrip("#")
        if len(hex_color) == 6:
            r = int(hex_color[0:2], 16)
            g = int(hex_color[2:4], 16)
            b = int(hex_color[4:6], 16)
            return f"{r}, {g}, {b}"
        return "0, 0, 0"

    def clear_log(self):
        """Clear all log entries"""
        self.log_entries = []
        self.filtered_entries = []
        self.log_text.clear()
        self.add_log_entry("INFO", "Log cleared")

    def export_log(self):
        """Export log entries to file"""
        from PySide6.QtWidgets import QFileDialog

        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export Log",
            f"cortex_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            "Text files (*.txt);;HTML files (*.html);;All files (*)",
        )

        if filename:
            try:
                if filename.endswith(".html"):
                    content = self._export_html()
                else:
                    content = self._export_text()

                with open(filename, "w", encoding="utf-8") as f:
                    f.write(content)

                self.add_log_entry("SUCCESS", f"Log exported to {filename}")
            except Exception as e:
                self.add_log_entry("ERROR", f"Export failed: {str(e)}")

    def _export_text(self) -> str:
        """Export as plain text"""
        lines = []
        for entry in self.log_entries:
            timestamp = entry["timestamp"].strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            lines.append(f"[{timestamp}] [{entry['type']}] {entry['message']}")
            if entry["details"]:
                # Indent details
                detail_lines = entry["details"].split("\n")
                for detail_line in detail_lines:
                    lines.append(f"    {detail_line}")
                lines.append("")  # Empty line after details

        return "\n".join(lines)

    def _export_html(self) -> str:
        """Export as HTML"""
        html_parts = ["<html><head><title>Cortex Processing Log</title></head><body>"]
        html_parts.append("<h1>Cortex Processing Log</h1>")
        html_parts.append(f"<p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>")

        for entry in self.log_entries:
            formatted_entry = self._format_log_entry(entry)
            html_parts.append(formatted_entry)

        html_parts.append("</body></html>")
        return "".join(html_parts)

    def update_analysis(self, analysis_data: Dict[str, Any]):
        """Update the analysis display with comprehensive information"""
        analysis = analysis_data.get("analysis")
        if not analysis:
            return

        # Update main analysis info
        self.reasoning_label.setText(analysis.reasoning)

        # Update metadata
        confidence = analysis.confidence
        self.confidence_label.setText(f"Confidence: {confidence:.1%}")

        # Color code confidence
        if confidence >= 0.8:
            color = "green"
        elif confidence >= 0.5:
            color = "orange"
        else:
            color = "red"
        self.confidence_label.setStyleSheet(f"font-weight: bold; color: {color};")

        self.query_type_label.setText(f"Type: {analysis.query_type}")
        self.complexity_label.setText(f"Complexity: {analysis.complexity}")

        # Update search actions summary
        if analysis.needs_search and analysis.search_actions:
            actions_text = f"Search Actions ({len(analysis.search_actions)}):\n"
            for i, action in enumerate(analysis.search_actions, 1):
                actions_text += f"{i}. {action.action_type.value}: {action.query[:50]}...\n"
        else:
            actions_text = "No search actions needed"

        self.search_actions_label.setText(actions_text)

        # Log the analysis details
        self.add_log_entry(
            "ANALYSIS",
            "Query analysis completed",
            f"Type: {analysis.query_type}, Confidence: {confidence:.1%}, Complexity: {analysis.complexity}",
        )

    def update_status(self, status: str):
        """Update the current status display"""
        self.status_label.setText(status)

        # Determine status color
        status_lower = status.lower()
        if "error" in status_lower or "failed" in status_lower:
            color = "#cc0000"
            bg_color = "#ffe6e6"
        elif "complete" in status_lower or "success" in status_lower:
            color = "#006600"
            bg_color = "#e6ffe6"
        elif "analyzing" in status_lower or "processing" in status_lower:
            color = "#cc6600"
            bg_color = "#fff5e6"
        else:
            color = "#0066cc"
            bg_color = "#f0f8ff"

        self.status_label.setStyleSheet(f"""
            QLabel {{
                color: {color};
                font-weight: bold;
                padding: 8px;
                background-color: {bg_color};
                border: 1px solid #ddd;
                border-radius: 4px;
            }}
        """)

        # Also log status changes
        if "error" in status_lower or "failed" in status_lower:
            log_type = "ERROR"
        elif "complete" in status_lower or "success" in status_lower:
            log_type = "SUCCESS"
        else:
            log_type = "INFO"

        self.add_log_entry(log_type, status)
