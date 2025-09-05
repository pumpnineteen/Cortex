import json
import aiohttp
from typing import List, Dict, Any, Optional
from PySide6.QtCore import Signal
from dataclasses import dataclass
from enum import Enum
import re
from datetime import datetime
import traceback

class SearchType(Enum):
    """Types of searches that can be performed"""

    WEB_SEARCH = "web_search"
    CODEBASE_SEARCH = "codebase_search"
    FILE_ANALYSIS = "file_analysis"
    RAG_SEARCH = "rag_search"  # Search through stored documents/knowledge base
    NONE = "none"


@dataclass
class SearchAction:
    """Represents a search action to be performed"""

    action_type: SearchType
    query: str
    priority: int = 1  # 1 = high, 2 = medium, 3 = low
    reasoning: str = ""  # Why this search is needed
    expected_info: str = ""  # What information we expect to find
    details: Optional[Dict[str, Any]] = None


@dataclass
class QueryAnalysisResult:
    """Result of query analysis"""

    original_query: str
    needs_search: bool
    search_actions: List[SearchAction]
    reasoning: str
    confidence: float  # 0.0 to 1.0
    query_type: str  # e.g., "factual", "how-to", "comparison", "current-events"
    complexity: str  # "simple", "medium", "complex"


class EnhancedLLMQueryAnalyzer:
    """LLM-powered query analyzer that uses reasoning instead of pattern matching"""
    log_entry = Signal(str, str, str)

    def __init__(self, config_manager, ollama_url="http://localhost:11434"):
        self.config_manager = config_manager
        self.ollama_url = ollama_url
        

    async def analyze_query(self, user_query: str, conversation_context: Optional[List[Dict]] = None) -> QueryAnalysisResult:
        """Aggressively determine search needs with proper QueryAnalysisResult format"""

        # Get the analysis model
        analysis_model = self.config_manager.get("ollama.models.search_extract", "") or self.config_manager.get(
            "ollama.models.chat", ""
        )

        if not analysis_model:
            return self._create_fallback_result(user_query, "No analysis model available")

        try:
            # Build comprehensive analysis prompt
            analysis_prompt = self._build_enhanced_analysis_prompt(user_query, conversation_context)
            self.log_entry.emit("INFO", "Analysis Prompt", analysis_prompt[:500] + "...")

            # Get analysis from LLM
            response = await self._call_ollama(analysis_model, analysis_prompt)
            self.log_entry.emit("INFO", "Analysis Response", response[:500] + "...")

            # Parse and validate the response
            return self._parse_enhanced_analysis_response(user_query, response)

        except Exception as e:
            full_traceback = traceback.format_exc()
            print(f"LLM Query analysis error: {e}")
            print(full_traceback)
            return self._create_fallback_result(user_query, f"Analysis failed: {str(e)}")

    def _build_enhanced_analysis_prompt(self, query: str, context: Optional[List[Dict]] = None) -> str:
        """Build a comprehensive analysis prompt that guides LLM reasoning"""

        # Build context from conversation history
        context_text = ""
        if context:
            recent_messages = context[-6:]  # Last 6 messages for better context
            for msg in recent_messages:
                role = msg.get("role", "")
                content = msg.get("content", "")[:300]  # Limit length
                context_text += f"{role}: {content}\n"

        current_date = datetime.now().strftime("%Y-%m-%d")

        prompt = f"""You are an expert query analysis system. Your job is to analyze user queries and determine what types of information gathering would provide the most helpful and complete response.

CURRENT DATE: {current_date}

CONVERSATION CONTEXT:
{context_text if context_text else "No previous conversation context"}

USER QUERY TO ANALYZE: "{query}"

AVAILABLE SEARCH TYPES:
1. WEB_SEARCH - Search internet for current information, news, tutorials, documentation, recent developments
2. CODEBASE_SEARCH - Search local code repositories for examples, functions, implementations
3. FILE_ANALYSIS - Analyze specific files, documents, or attachments mentioned
4. RAG_SEARCH - Search through stored knowledge base, documentation, or previous conversations
5. NONE - Can be answered adequately from your existing knowledge

ANALYSIS FRAMEWORK:
Please reason through the following steps:

1. QUERY CLASSIFICATION:
   - What type of query is this? (factual, how-to, comparison, troubleshooting, current-events, creative, etc.)
   - What is the complexity level? (simple, medium, complex)
   - What domain does it relate to? (tech, science, current events, programming, etc.)

2. INFORMATION NEEDS ASSESSMENT:
   - What specific information is needed to answer this well?
   - Is current/recent information crucial? (last days/weeks/months)
   - Are there technical details that need verification?
   - Would code examples or implementations be helpful?
   - Are there multiple perspectives or sources needed?

3. SEARCH STRATEGY:
   - Which search types would be most valuable?
   - What specific queries would find the needed information?
   - What's the priority order of searches?
   - What information do you expect each search to provide?

4. CONFIDENCE ASSESSMENT:
   - How confident are you in this analysis?
   - Are there ambiguities or edge cases to consider?

Respond with a JSON object in this EXACT format (no additional text):
{{
    "needs_search": true/false,
    "query_type": "classification of the query type",
    "complexity": "simple|medium|complex", 
    "reasoning": "detailed reasoning for your analysis decisions",
    "confidence": 0.0-1.0,
    "actions": [
        {{
            "type": "WEB_SEARCH|CODEBASE_SEARCH|FILE_ANALYSIS|RAG_SEARCH",
            "query": "specific optimized search query",
            "priority": 1-3,
            "reasoning": "why this specific search is needed", 
            "expected_info": "what information this search should provide"
        }}
    ]
}}

GUIDELINES FOR GOOD ANALYSIS:
- Current events, news, recent developments → WEB_SEARCH with high priority
- Programming questions → Consider both WEB_SEARCH (docs/tutorials) AND CODEBASE_SEARCH
- "How to" questions → WEB_SEARCH for tutorials and guides
- Factual questions about recent info → WEB_SEARCH
- Questions about specific files/documents → FILE_ANALYSIS
- Complex topics → Multiple complementary searches
- Simple definitional questions that don't need current info → No search needed
- Creative tasks → Usually no search needed unless research required

SEARCH QUERY OPTIMIZATION:
- Make search queries specific and focused
- Include relevant technical terms and context
- For programming: include language/framework names
- For current events: include timeframe indicators
- Avoid overly broad or vague search terms

Remember: The goal is to gather information that will enable providing the most helpful, accurate, and complete response possible."""

        return prompt

    async def _call_ollama(self, model: str, prompt: str) -> str:
        """Enhanced Ollama API call with better parameters for analysis tasks"""

        messages = [{"role": "user", "content": prompt}]
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": 0.1,  # Low temperature for consistent analysis
                "top_k": 10,
                "top_p": 0.3,
                "repeat_penalty": 1.1,
                "stop": ["\n\n\n"],  # Stop at multiple newlines
            },
        }

        timeout = aiohttp.ClientTimeout(total=30)  # 30 second timeout

        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(f"{self.ollama_url}/api/chat", json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    return data["message"]["content"]
                else:
                    error_text = await response.text()
                    raise Exception(f"Ollama API error {response.status}: {error_text}")

    def _parse_enhanced_analysis_response(self, original_query: str, response: str) -> QueryAnalysisResult:
        """Parse and validate the enhanced JSON response from the analysis model"""

        try:
            # Clean and extract JSON
            json_str = self._extract_json_from_response(response)
            data = json.loads(json_str)

            # Validate required fields
            self._validate_analysis_response(data)

            # Build search actions with enhanced information
            search_actions = []
            for action_data in data.get("actions", []):
                try:
                    action_type = SearchType(action_data.get("type").lower())

                    search_action = SearchAction(
                        action_type=action_type,
                        query=action_data.get("query", "").strip(),
                        priority=int(action_data.get("priority", 2)),
                        reasoning=action_data.get("reasoning", "").strip(),
                        expected_info=action_data.get("expected_info", "").strip(),
                        details=action_data.get("details", {}),
                    )

                    # Validate search action
                    if search_action.query and search_action.action_type != SearchType.NONE:
                        search_actions.append(search_action)

                except (ValueError, KeyError) as e:
                    print(f"Skipping invalid search action: {e}")
                    continue

            return QueryAnalysisResult(
                original_query=original_query,
                needs_search=bool(data.get("needs_search", False) and search_actions),
                search_actions=search_actions,
                reasoning=data.get("reasoning", "").strip(),
                confidence=float(data.get("confidence", 0.5)),
                query_type=data.get("query_type", "unknown").strip(),
                complexity=data.get("complexity", "medium").strip(),
            )

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            print(f"JSON parsing error in enhanced analysis: {e}")
            print(f"Raw response: {response[:500]}...")
            return self._create_intelligent_fallback(original_query, response)

    def _extract_json_from_response(self, response: str) -> str:
        """Extract JSON object from response, handling various formatting issues"""

        # Remove markdown code blocks if present
        response = re.sub(r"```json\s*", "", response)
        response = re.sub(r"```\s*$", "", response.strip())

        # Find JSON object boundaries
        json_start = response.find("{")
        if json_start == -1:
            raise ValueError("No JSON object found in response")

        # Find matching closing brace
        brace_count = 0
        json_end = -1

        for i, char in enumerate(response[json_start:], json_start):
            if char == "{":
                brace_count += 1
            elif char == "}":
                brace_count -= 1
                if brace_count == 0:
                    json_end = i + 1
                    break

        if json_end == -1:
            json_end = response.rfind("}") + 1

        if json_end <= json_start:
            raise ValueError("Could not find complete JSON object")

        return response[json_start:json_end]

    def _validate_analysis_response(self, data: Dict[str, Any]):
        """Validate that the analysis response has required fields"""

        required_fields = ["needs_search", "reasoning", "confidence"]
        for field in required_fields:
            if field not in data:
                raise KeyError(f"Missing required field: {field}")

        # Validate confidence range
        confidence = data.get("confidence", 0.5)
        if not isinstance(confidence, (int, float)) or confidence < 0 or confidence > 1:
            data["confidence"] = 0.5  # Default to medium confidence

        # Validate actions format
        if "actions" in data and not isinstance(data["actions"], list):
            raise ValueError("Actions must be a list")

    def _create_intelligent_fallback(self, query: str, response: str) -> QueryAnalysisResult:
        """Create a more intelligent fallback analysis using heuristics"""

        query_lower = query.lower()
        actions = []

        # Analyze query characteristics
        is_current_info = any(
            term in query_lower
            for term in ["latest", "recent", "current", "new", "today", "now", "2024", "2025", "update", "news", "recent"]
        )

        is_programming = any(
            term in query_lower
            for term in [
                "code",
                "function",
                "class",
                "debug",
                "error",
                "python",
                "javascript",
                "java",
                "c++",
                "programming",
                "api",
                "library",
                "framework",
                "syntax",
                "algorithm",
                "implementation",
            ]
        )

        is_how_to = any(
            phrase in query_lower
            for phrase in [
                "how to",
                "how do",
                "how can",
                "tutorial",
                "guide",
                "steps",
                "instructions",
                "example",
                "demonstrate",
            ]
        )

        is_comparison = any(
            term in query_lower
            for term in ["vs", "versus", "compare", "difference", "better", "best", "advantages", "pros and cons"]
        )

        # Build actions based on characteristics
        if is_current_info:
            actions.append(
                SearchAction(
                    SearchType.WEB_SEARCH,
                    query,
                    priority=1,
                    reasoning="Query requires current/recent information",
                    expected_info="Latest information and developments",
                )
            )

        if is_programming:
            if is_how_to:
                actions.append(
                    SearchAction(
                        SearchType.WEB_SEARCH,
                        f"{query} tutorial documentation examples",
                        priority=1,
                        reasoning="Programming how-to requires documentation and examples",
                        expected_info="Tutorials, documentation, and code examples",
                    )
                )
            actions.append(
                SearchAction(
                    SearchType.CODEBASE_SEARCH,
                    f"code examples {query}",
                    priority=2,
                    reasoning="Programming query benefits from code examples",
                    expected_info="Relevant code implementations and patterns",
                )
            )

        elif is_how_to and not is_programming:
            actions.append(
                SearchAction(
                    SearchType.WEB_SEARCH,
                    f"{query} guide tutorial steps",
                    priority=1,
                    reasoning="How-to query requires step-by-step instructions",
                    expected_info="Detailed guides and tutorials",
                )
            )

        if is_comparison:
            actions.append(
                SearchAction(
                    SearchType.WEB_SEARCH,
                    query,
                    priority=1,
                    reasoning="Comparison requires up-to-date information from multiple sources",
                    expected_info="Comparative analysis and current information",
                )
            )

        # Determine if we need search based on query length and characteristics
        needs_search = (
            len(actions) > 0
            or len(query.split()) > 5  # Longer queries often need more info
            or is_current_info
            or is_how_to
            or is_comparison
        )

        # If no specific actions but seems like it needs search, add general web search
        if needs_search and not actions:
            actions.append(
                SearchAction(
                    SearchType.WEB_SEARCH,
                    query,
                    priority=2,
                    reasoning="Complex query that would benefit from additional information",
                    expected_info="Comprehensive information to supplement knowledge",
                )
            )

        return QueryAnalysisResult(
            original_query=query,
            needs_search=needs_search,
            search_actions=actions,
            reasoning=f"Intelligent fallback analysis based on query characteristics. Response parsing failed: {response[:100]}...",
            confidence=0.4,  # Lower confidence for fallback
            query_type="unknown",
            complexity="medium",
        )

    def _create_fallback_result(self, query: str, error_msg: str) -> QueryAnalysisResult:
        """Create a basic fallback result when analysis completely fails"""

        return QueryAnalysisResult(
            original_query=query,
            needs_search=False,
            search_actions=[],
            reasoning=error_msg,
            confidence=0.0,
            query_type="error",
            complexity="unknown",
        )


# Enhanced Chat Manager that uses the new analyzer
class EnhancedIterativeChatManager:
    """Enhanced chat manager using LLM-based query analysis"""

    def __init__(self, config_manager, status_callback=None):
        self.config_manager = config_manager
        self.query_analyzer = EnhancedLLMQueryAnalyzer(config_manager)
        self.status_callback = status_callback

    async def process_query(self, user_query: str, conversation_context: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """Process a user query through enhanced iterative analysis"""

        self._update_status("Analyzing query with LLM reasoning...")

        # Step 1: Enhanced LLM-based analysis
        analysis = await self.query_analyzer.analyze_query(user_query, conversation_context)

        self._update_status(f"Analysis complete. Type: {analysis.query_type}, Complexity: {analysis.complexity}")

        # Step 2: Execute searches with detailed reasoning
        search_results = {}
        if analysis.needs_search:
            self._update_status(f"Executing {len(analysis.search_actions)} strategic search actions...")

            for i, action in enumerate(analysis.search_actions, 1):
                self._update_status(f"Step {i}: {action.action_type.value} - {action.reasoning}")

                # Execute the search (placeholder - would be implemented based on search type)
                result = await self._execute_search_action(action)
                search_results[f"search_{i}"] = {
                    "action": action,
                    "result": result,
                    "reasoning": action.reasoning,
                    "expected_info": action.expected_info,
                }

        # Step 3: Generate response with enhanced context
        self._update_status("Generating comprehensive response...")
        final_response = await self._generate_enhanced_response(user_query, analysis, search_results)

        self._update_status("Analysis and response complete")

        return {
            "analysis": analysis,
            "search_results": search_results,
            "response": final_response,
            "steps_performed": self._format_enhanced_steps(analysis, search_results),
            "metadata": {
                "query_type": analysis.query_type,
                "complexity": analysis.complexity,
                "confidence": analysis.confidence,
                "searches_performed": len(search_results),
            },
        }

    async def _execute_search_action(self, action: SearchAction) -> Dict[str, Any]:
        """Execute search action with enhanced metadata"""

        # This would be implemented with actual search functionality
        return {
            "type": action.action_type.value,
            "status": "not_implemented",
            "query": action.query,
            "reasoning": action.reasoning,
            "expected_info": action.expected_info,
        }

    async def _generate_enhanced_response(self, user_query: str, analysis: QueryAnalysisResult, search_results: Dict) -> str:
        """Generate response with enhanced context and reasoning"""

        chat_model = self.config_manager.get("ollama.models.chat", "")
        if not chat_model:
            return "No chat model configured"

        # Build rich context from analysis and search results
        context = f"""
Query Analysis:
- Type: {analysis.query_type}
- Complexity: {analysis.complexity}  
- Reasoning: {analysis.reasoning}
- Confidence: {analysis.confidence:.2f}

Search Strategy Executed:
"""

        for key, result in search_results.items():
            action = result["action"]
            context += f"""
- {action.action_type.value}: {action.query}
  Reasoning: {action.reasoning}
  Expected: {action.expected_info}
  Status: {result["result"].get("status", "unknown")}
"""

        prompt = f"""You are providing a comprehensive response based on detailed query analysis and strategic information gathering.

User Query: {user_query}

{context}

Based on this thorough analysis and the search strategy that was executed, provide a helpful, detailed response that directly addresses the user's question. If the searches were not implemented yet, acknowledge this limitation and provide the best possible response from your existing knowledge while explaining what additional information would have been gathered."""

        try:
            response = await self.query_analyzer._call_ollama(chat_model, prompt)
            return response
        except Exception as e:
            return f"Error generating enhanced response: {str(e)}"

    def _update_status(self, status: str):
        """Update status via callback"""
        if self.status_callback:
            self.status_callback(status)

    def _format_enhanced_steps(self, analysis: QueryAnalysisResult, search_results: Dict) -> str:
        """Format enhanced steps with reasoning"""

        steps = [
            "1. LLM Query Analysis:",
            f"   - Classification: {analysis.query_type} ({analysis.complexity} complexity)",
            f"   - Reasoning: {analysis.reasoning}",
            f"   - Confidence: {analysis.confidence:.2f}",
        ]

        if analysis.needs_search:
            steps.append(f"2. Strategic Search Actions: {len(analysis.search_actions)} planned")
            for i, (key, result) in enumerate(search_results.items(), 1):
                action = result["action"]
                steps.append(f"   {i}. {action.action_type.value}:")
                steps.append(f"      Query: {action.query}")
                steps.append(f"      Reasoning: {action.reasoning}")
                steps.append(f"      Status: {result['result'].get('status', 'unknown')}")
        else:
            steps.append("2. No additional search needed - can answer from existing knowledge")

        steps.append("3. Generated comprehensive response")
        return "\n".join(steps)