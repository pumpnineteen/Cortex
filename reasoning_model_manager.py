import re
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class ReasoningFormat(Enum):
    """Known reasoning formats used by different models"""

    THINKING_TAGS = "thinking_tags"  # <thinking>...</thinking>
    REASONING_TAGS = "reasoning_tags"  # <reasoning>...</reasoning>
    COT_MARKERS = "cot_markers"  # "Let me think..." or "Step by step:"
    REFLECTION_TAGS = "reflection_tags"  # <reflection>...</reflection>
    ANALYSIS_TAGS = "analysis_tags"  # <analysis>...</analysis>
    CUSTOM = "custom"
    NONE = "none"


@dataclass
class ReasoningModelConfig:
    """Configuration for a reasoning model"""

    model_name: str
    is_reasoning: bool
    format: ReasoningFormat
    custom_pattern: Optional[str] = None
    prompt_suffix: Optional[str] = None


class ReasoningModelManager:
    """Manages reasoning model detection and configuration"""

    # Known reasoning models and their typical formats
    KNOWN_REASONING_MODELS = {
        # Qwen reasoning models
        r"qwen.*r1": ReasoningFormat.THINKING_TAGS,
        r"qwen.*reasoning": ReasoningFormat.THINKING_TAGS,
        r"qwen.*cot": ReasoningFormat.COT_MARKERS,
        # DeepSeek reasoning models
        r"deepseek.*r1": ReasoningFormat.THINKING_TAGS,
        r"deepseek.*reasoning": ReasoningFormat.THINKING_TAGS,
        # Other reasoning models
        r".*reasoning.*": ReasoningFormat.THINKING_TAGS,
        r".*cot.*": ReasoningFormat.COT_MARKERS,
        r".*r1.*": ReasoningFormat.THINKING_TAGS,
        # OpenAI o1 style models
        r"o1.*": ReasoningFormat.THINKING_TAGS,
        # Claude reasoning variants (hypothetical)
        r"claude.*reasoning": ReasoningFormat.ANALYSIS_TAGS,
    }

    def __init__(self, config_manager):
        self.config_manager = config_manager
        self.model_configs = self._load_reasoning_configs()

    def _load_reasoning_configs(self) -> Dict[str, ReasoningModelConfig]:
        """Load reasoning model configurations from config"""
        configs = {}

        # Load from config file
        reasoning_config = self.config_manager.get("reasoning_models", {})

        for model_name, config_data in reasoning_config.items():
            configs[model_name] = ReasoningModelConfig(
                model_name=model_name,
                is_reasoning=config_data.get("is_reasoning", False),
                format=ReasoningFormat(config_data.get("format", "none")),
                custom_pattern=config_data.get("custom_pattern"),
                prompt_suffix=config_data.get("prompt_suffix"),
            )

        return configs

    def is_reasoning_model(self, model_name: str) -> bool:
        """Check if a model is configured as reasoning model"""
        # Check explicit configuration first
        if model_name in self.model_configs:
            return self.model_configs[model_name].is_reasoning

        # Check against known patterns
        model_lower = model_name.lower()
        for pattern in self.KNOWN_REASONING_MODELS:
            if re.search(pattern, model_lower):
                return True

        return False

    def get_reasoning_format(self, model_name: str) -> ReasoningFormat:
        """Get the reasoning format for a model"""
        # Check explicit configuration
        if model_name in self.model_configs:
            return self.model_configs[model_name].format

        # Check known patterns
        model_lower = model_name.lower()
        for pattern, format_type in self.KNOWN_REASONING_MODELS.items():
            if re.search(pattern, model_lower):
                return format_type

        return ReasoningFormat.NONE

    def get_reasoning_prompt_suffix(self, model_name: str) -> Optional[str]:
        """Get prompt suffix to encourage reasoning format"""
        config = self.model_configs.get(model_name)
        if config and config.prompt_suffix:
            return config.prompt_suffix

        format_type = self.get_reasoning_format(model_name)

        # Default prompt suffixes for different formats
        suffix_map = {
            ReasoningFormat.THINKING_TAGS: "\n\nPlease show your reasoning using <thinking>...</thinking> tags before providing your final answer.",
            ReasoningFormat.REASONING_TAGS: "\n\nPlease show your reasoning using <reasoning>...</reasoning> tags before providing your final answer.",
            ReasoningFormat.COT_MARKERS: "\n\nPlease think through this step by step.",
            ReasoningFormat.REFLECTION_TAGS: "\n\nPlease reflect on your reasoning using <reflection>...</reflection> tags.",
            ReasoningFormat.ANALYSIS_TAGS: "\n\nPlease show your analysis using <analysis>...</analysis> tags.",
        }

        return suffix_map.get(format_type)

    def configure_model(
        self,
        model_name: str,
        is_reasoning: bool,
        format: ReasoningFormat,
        custom_pattern: Optional[str] = None,
        prompt_suffix: Optional[str] = None,
    ):
        """Configure a model's reasoning settings"""
        config = ReasoningModelConfig(
            model_name=model_name,
            is_reasoning=is_reasoning,
            format=format,
            custom_pattern=custom_pattern,
            prompt_suffix=prompt_suffix,
        )

        self.model_configs[model_name] = config
        self._save_reasoning_configs()

    def _save_reasoning_configs(self):
        """Save reasoning configurations to config file"""
        reasoning_config = {}

        for model_name, config in self.model_configs.items():
            reasoning_config[model_name] = {
                "is_reasoning": config.is_reasoning,
                "format": config.format.value,
                "custom_pattern": config.custom_pattern,
                "prompt_suffix": config.prompt_suffix,
            }

        self.config_manager.set("reasoning_models", reasoning_config)
        self.config_manager.save_config()

    def auto_detect_reasoning_format(self, model_name: str, response: str) -> Optional[ReasoningFormat]:
        """Attempt to auto-detect reasoning format from a response"""
        response_lower = response.lower()

        # Check for various reasoning patterns
        if re.search(r"<thinking>.*?</thinking>", response, re.DOTALL | re.IGNORECASE):
            return ReasoningFormat.THINKING_TAGS
        elif re.search(r"<reasoning>.*?</reasoning>", response, re.DOTALL | re.IGNORECASE):
            return ReasoningFormat.REASONING_TAGS
        elif re.search(r"<reflection>.*?</reflection>", response, re.DOTALL | re.IGNORECASE):
            return ReasoningFormat.REFLECTION_TAGS
        elif re.search(r"<analysis>.*?</analysis>", response, re.DOTALL | re.IGNORECASE):
            return ReasoningFormat.ANALYSIS_TAGS
        elif re.search(r"(let me think|step by step|first,|second,|finally)", response_lower):
            return ReasoningFormat.COT_MARKERS

        return None

    def learn_from_response(self, model_name: str, response: str):
        """Learn reasoning format from a model's response"""
        if model_name in self.model_configs:
            return  # Already configured

        detected_format = self.auto_detect_reasoning_format(model_name, response)
        if detected_format:
            # Auto-configure the model
            self.configure_model(model_name=model_name, is_reasoning=True, format=detected_format)
            print(f"Auto-detected reasoning format for {model_name}: {detected_format.value}")


class ReasoningParser:
    """Parser for extracting reasoning chains from model responses"""

    def __init__(self, reasoning_manager: ReasoningModelManager):
        self.reasoning_manager = reasoning_manager

    def parse_response(self, model_name: str, response: str) -> Tuple[Optional[str], str]:
        """Parse response into reasoning and final answer

        Returns:
            Tuple of (reasoning_text, final_answer)
        """
        if not self.reasoning_manager.is_reasoning_model(model_name):
            return None, response

        format_type = self.reasoning_manager.get_reasoning_format(model_name)

        if format_type == ReasoningFormat.THINKING_TAGS:
            return self._parse_thinking_tags(response)
        elif format_type == ReasoningFormat.REASONING_TAGS:
            return self._parse_reasoning_tags(response)
        elif format_type == ReasoningFormat.REFLECTION_TAGS:
            return self._parse_reflection_tags(response)
        elif format_type == ReasoningFormat.ANALYSIS_TAGS:
            return self._parse_analysis_tags(response)
        elif format_type == ReasoningFormat.COT_MARKERS:
            return self._parse_cot_markers(response)
        elif format_type == ReasoningFormat.CUSTOM:
            return self._parse_custom_pattern(model_name, response)

        return None, response

    def _parse_thinking_tags(self, response: str) -> Tuple[Optional[str], str]:
        """Parse <thinking>...</thinking> tags"""
        thinking_pattern = r"<thinking>(.*?)</thinking>"
        matches = re.findall(thinking_pattern, response, re.DOTALL | re.IGNORECASE)

        if matches:
            thinking_text = "\n\n".join(matches)
            # Remove thinking tags from final answer
            final_answer = re.sub(thinking_pattern, "", response, flags=re.DOTALL | re.IGNORECASE)
            final_answer = final_answer.strip()
            return thinking_text.strip(), final_answer

        return None, response

    def _parse_reasoning_tags(self, response: str) -> Tuple[Optional[str], str]:
        """Parse <reasoning>...</reasoning> tags"""
        reasoning_pattern = r"<reasoning>(.*?)</reasoning>"
        matches = re.findall(reasoning_pattern, response, re.DOTALL | re.IGNORECASE)

        if matches:
            reasoning_text = "\n\n".join(matches)
            final_answer = re.sub(reasoning_pattern, "", response, flags=re.DOTALL | re.IGNORECASE)
            final_answer = final_answer.strip()
            return reasoning_text.strip(), final_answer

        return None, response

    def _parse_reflection_tags(self, response: str) -> Tuple[Optional[str], str]:
        """Parse <reflection>...</reflection> tags"""
        reflection_pattern = r"<reflection>(.*?)</reflection>"
        matches = re.findall(reflection_pattern, response, re.DOTALL | re.IGNORECASE)

        if matches:
            reflection_text = "\n\n".join(matches)
            final_answer = re.sub(reflection_pattern, "", response, flags=re.DOTALL | re.IGNORECASE)
            final_answer = final_answer.strip()
            return reflection_text.strip(), final_answer

        return None, response

    def _parse_analysis_tags(self, response: str) -> Tuple[Optional[str], str]:
        """Parse <analysis>...</analysis> tags"""
        analysis_pattern = r"<analysis>(.*?)</analysis>"
        matches = re.findall(analysis_pattern, response, re.DOTALL | re.IGNORECASE)

        if matches:
            analysis_text = "\n\n".join(matches)
            final_answer = re.sub(analysis_pattern, "", response, flags=re.DOTALL | re.IGNORECASE)
            final_answer = final_answer.strip()
            return analysis_text.strip(), final_answer

        return None, response

    def _parse_cot_markers(self, response: str) -> Tuple[Optional[str], str]:
        """Parse chain-of-thought markers"""
        # This is more heuristic - look for common CoT patterns
        lines = response.split("\n")
        thinking_lines = []
        answer_lines = []

        in_thinking = False
        thinking_triggers = ["let me think", "step by step", "first,", "to solve this"]
        answer_triggers = ["therefore", "in conclusion", "the answer is", "final answer"]

        for line in lines:
            line_lower = line.lower().strip()

            if any(trigger in line_lower for trigger in thinking_triggers):
                in_thinking = True
            elif any(trigger in line_lower for trigger in answer_triggers):
                in_thinking = False

            if in_thinking:
                thinking_lines.append(line)
            else:
                answer_lines.append(line)

        if thinking_lines:
            return "\n".join(thinking_lines).strip(), "\n".join(answer_lines).strip()

        return None, response

    def _parse_custom_pattern(self, model_name: str, response: str) -> Tuple[Optional[str], str]:
        """Parse custom pattern defined for specific model"""
        config = self.reasoning_manager.model_configs.get(model_name)
        if not config or not config.custom_pattern:
            return None, response

        try:
            matches = re.findall(config.custom_pattern, response, re.DOTALL | re.IGNORECASE)
            if matches:
                reasoning_text = "\n\n".join(matches)
                final_answer = re.sub(config.custom_pattern, "", response, flags=re.DOTALL | re.IGNORECASE)
                return reasoning_text.strip(), final_answer.strip()
        except re.error:
            print(f"Invalid custom pattern for {model_name}: {config.custom_pattern}")

        return None, response
