import json
import asyncio
import aiohttp
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum


class SearchType(Enum):
    """Types of searches that can be performed"""

    WEB_SEARCH = "web_search"
    CODEBASE_SEARCH = "codebase_search"
    FILE_ANALYSIS = "file_analysis"
    NONE = "none"


@dataclass
class SearchAction:
    """Represents a search action to be performed"""

    action_type: SearchType
    query: str
    priority: int = 1  # 1 = high, 2 = medium, 3 = low
    details: Optional[Dict[str, Any]] = None


@dataclass
class QueryAnalysisResult:
    """Result of query analysis"""

    original_query: str
    needs_search: bool
    search_actions: List[SearchAction]
    reasoning: str
    confidence: float  # 0.0 to 1.0


class QueryAnalyzer:
    """Analyzes queries to determine what searches/actions are needed"""

    def __init__(self, config_manager, ollama_url="http://localhost:11434"):
        self.config_manager = config_manager
        self.ollama_url = ollama_url

    async def analyze_query(self, user_query: str, conversation_context: Optional[List[Dict]] = None) -> QueryAnalysisResult:
        """Analyze a user query to determine what searches are needed"""

        # Build context from conversation history
        context_text = ""
        if conversation_context:
            recent_messages = conversation_context[-4:]  # Last 4 messages
            for msg in recent_messages:
                role = msg.get("role", "")
                content = msg.get("content", "")
                context_text += f"{role}: {content[:200]}...\n"

        # Get the analysis model
        analysis_model = self.config_manager.get("ollama.models.search_extract", "") or self.config_manager.get(
            "ollama.models.chat", ""
        )

        if not analysis_model:
            return QueryAnalysisResult(
                original_query=user_query,
                needs_search=False,
                search_actions=[],
                reasoning="No analysis model available",
                confidence=0.0,
            )

        # Construct analysis prompt
        analysis_prompt = self._build_analysis_prompt(user_query, context_text)

        try:
            # Send to Ollama for analysis
            response = await self._call_ollama(analysis_model, analysis_prompt)

            # Parse the response
            return self._parse_analysis_response(user_query, response)

        except Exception as e:
            print(f"Query analysis error: {e}")
            return QueryAnalysisResult(
                original_query=user_query,
                needs_search=False,
                search_actions=[],
                reasoning=f"Analysis failed: {str(e)}",
                confidence=0.0,
            )

    def _build_analysis_prompt(self, query: str, context: str = "") -> str:
        """Build the prompt for query analysis"""

        base_prompt = self.config_manager.get(
            "prompts.search_extract",
            "Extract relevant information from the following web search results for the query: {query}",
        )

        # Create a specialized analysis prompt
        prompt = f"""You are a query analysis assistant. Analyze the following user query and determine what types of searches or information gathering would be helpful to provide a complete answer.

Context from recent conversation:
{context}

User Query: "{query}"

Available search types:
1. WEB_SEARCH - Search the internet for current information, news, facts, tutorials, etc.
2. CODEBASE_SEARCH - Search through local code repositories for relevant code examples, functions, classes
3. FILE_ANALYSIS - Analyze attached files, documents, or specific local files
4. NONE - No additional search needed, can answer from general knowledge

Please respond with a JSON object in this exact format:
{{
    "needs_search": true/false,
    "reasoning": "brief explanation of your analysis",
    "confidence": 0.0-1.0,
    "actions": [
        {{
            "type": "WEB_SEARCH|CODEBASE_SEARCH|FILE_ANALYSIS",
            "query": "specific search query",
            "priority": 1-3,
            "details": {{"any": "additional info"}}
        }}
    ]
}}

Guidelines:
- If the query is about current events, news, or recent information, suggest WEB_SEARCH
- If asking about programming concepts, code examples, or debugging, consider CODEBASE_SEARCH
- If the query references specific files or documents, suggest FILE_ANALYSIS
- For general knowledge questions that don't need current info, set needs_search to false
- Priority: 1=high, 2=medium, 3=low
- Keep search queries focused and specific
- Maximum 3 search actions per query"""

        return prompt

    async def _call_ollama(self, model: str, prompt: str) -> str:
        """Call Ollama API with the analysis prompt"""

        messages = [{"role": "user", "content": prompt}]
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": 0.1,  # Low temperature for more consistent analysis
                "top_k": 10,
                "top_p": 0.3,
            },
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(f"{self.ollama_url}/api/chat", json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    return data["message"]["content"]
                else:
                    raise Exception(f"Ollama API error: {response.status}")

    def _parse_analysis_response(self, original_query: str, response: str) -> QueryAnalysisResult:
        """Parse the JSON response from the analysis model"""

        try:
            # Try to extract JSON from response
            json_start = response.find("{")
            json_end = response.rfind("}") + 1

            if json_start == -1 or json_end == 0:
                raise ValueError("No JSON found in response")

            json_str = response[json_start:json_end]
            data = json.loads(json_str)

            # Build search actions
            search_actions = []
            for action_data in data.get("actions", []):
                action_type = SearchType(action_data.get("type", "NONE"))
                search_actions.append(
                    SearchAction(
                        action_type=action_type,
                        query=action_data.get("query", ""),
                        priority=action_data.get("priority", 2),
                        details=action_data.get("details", {}),
                    )
                )

            return QueryAnalysisResult(
                original_query=original_query,
                needs_search=data.get("needs_search", False),
                search_actions=search_actions,
                reasoning=data.get("reasoning", ""),
                confidence=float(data.get("confidence", 0.5)),
            )

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            # Fallback: try to determine search need from keywords
            return self._fallback_analysis(original_query, response)

    def _fallback_analysis(self, query: str, response: str) -> QueryAnalysisResult:
        """Fallback analysis when JSON parsing fails"""

        query_lower = query.lower()
        needs_web_search = any(
            keyword in query_lower
            for keyword in [
                "latest",
                "current",
                "recent",
                "news",
                "today",
                "update",
                "what's new",
                "happening now",
                "2024",
                "2025",
            ]
        )

        needs_code_search = any(
            keyword in query_lower
            for keyword in [
                "code",
                "function",
                "class",
                "import",
                "debug",
                "error",
                "python",
                "javascript",
                "java",
                "c++",
                "programming",
            ]
        )

        actions = []
        if needs_web_search:
            actions.append(SearchAction(SearchType.WEB_SEARCH, query, 1))
        if needs_code_search:
            actions.append(SearchAction(SearchType.CODEBASE_SEARCH, query, 1))

        return QueryAnalysisResult(
            original_query=query,
            needs_search=len(actions) > 0,
            search_actions=actions,
            reasoning=f"Fallback analysis from response: {response[:100]}...",
            confidence=0.3,
        )


class IterativeChatManager:
    """Manages the iterative chat process with search integration"""

    def __init__(self, config_manager, status_callback=None):
        self.config_manager = config_manager
        self.query_analyzer = QueryAnalyzer(config_manager)
        self.status_callback = status_callback  # Callback to update UI with status

    async def process_query(self, user_query: str, conversation_context: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """Process a user query through the iterative chat system"""

        self._update_status("Analyzing query...")

        # Step 1: Analyze the query
        analysis = await self.query_analyzer.analyze_query(user_query, conversation_context)

        self._update_status(f"Analysis complete. Reasoning: {analysis.reasoning}")

        # Step 2: Execute searches if needed
        search_results = {}
        if analysis.needs_search:
            self._update_status(f"Executing {len(analysis.search_actions)} search actions...")

            for i, action in enumerate(analysis.search_actions, 1):
                self._update_status(f"Step {i}: {action.action_type.value} - {action.query}")

                # This is where we'll call the specific search implementations
                result = await self._execute_search_action(action)
                search_results[f"search_{i}"] = {"action": action, "result": result}

        # Step 3: Generate final response (using chat model)
        self._update_status("Generating response...")
        final_response = await self._generate_final_response(user_query, analysis, search_results)

        self._update_status("Complete")

        return {
            "analysis": analysis,
            "search_results": search_results,
            "response": final_response,
            "steps_performed": self._format_steps(analysis, search_results),
        }

    async def _execute_search_action(self, action: SearchAction) -> Dict[str, Any]:
        """Execute a specific search action (placeholder for now)"""

        if action.action_type == SearchType.WEB_SEARCH:
            return {"type": "web", "status": "not_implemented", "query": action.query}
        elif action.action_type == SearchType.CODEBASE_SEARCH:
            return {"type": "codebase", "status": "not_implemented", "query": action.query}
        elif action.action_type == SearchType.FILE_ANALYSIS:
            return {"type": "file", "status": "not_implemented", "query": action.query}
        else:
            return {"type": "unknown", "status": "skipped"}

    async def _generate_final_response(
        self, user_query: str, analysis: QueryAnalysisResult, search_results: Dict[str, Any]
    ) -> str:
        """Generate the final response using the chat model"""

        chat_model = self.config_manager.get("ollama.models.chat", "")
        if not chat_model:
            return "No chat model configured"

        # Build context from search results
        context = ""
        for key, result in search_results.items():
            action = result["action"]
            context += f"\n{action.action_type.value}: {action.query}\n"
            context += f"Result: {result['result']}\n"

        prompt = f"""Based on the user's query and the following search results, provide a helpful response.

User Query: {user_query}

Analysis Reasoning: {analysis.reasoning}

Search Results:
{context}

Please provide a comprehensive answer that incorporates the relevant information from the searches."""

        try:
            analyzer = QueryAnalyzer(self.config_manager)
            response = await analyzer._call_ollama(chat_model, prompt)
            return response
        except Exception as e:
            return f"Error generating response: {str(e)}"

    def _update_status(self, status: str):
        """Update status via callback"""
        if self.status_callback:
            self.status_callback(status)
        else:
            print(f"Status: {status}")

    def _format_steps(self, analysis: QueryAnalysisResult, search_results: Dict[str, Any]) -> str:
        """Format the steps performed for display"""
        steps = [f"1. Query Analysis: {analysis.reasoning}"]

        if analysis.needs_search:
            steps.append(f"2. Search Actions Required: {len(analysis.search_actions)}")
            for i, (key, result) in enumerate(search_results.items(), 1):
                action = result["action"]
                steps.append(f"   {i}. {action.action_type.value}: {action.query}")

        steps.append("3. Generated final response")
        return "\n".join(steps)
