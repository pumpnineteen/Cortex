import hashlib
import json
import sqlite3
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from query_analyzer import QueryAnalysisResult, SearchAction, SearchType


@dataclass
class CachedSearchResult:
    """Cached search result with metadata"""

    query_hash: str
    search_type: str
    query: str
    results: Dict[str, Any]
    timestamp: float
    expires_at: float
    source: str


class SearchCache:
    """SQLite-based search result cache"""

    def __init__(self, config_manager):
        self.config_manager = config_manager
        self.cache_dir = Path.home() / ".cortex" / "cache"
        self.cache_dir.mkdir(exist_ok=True)
        self.db_path = self.cache_dir / "search_cache.db"
        self._init_db()

    def _init_db(self):
        """Initialize cache database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS search_cache (
                    query_hash TEXT PRIMARY KEY,
                    search_type TEXT,
                    original_query TEXT,
                    results TEXT,
                    timestamp REAL,
                    expires_at REAL,
                    source TEXT
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_expires ON search_cache(expires_at)")

    def _hash_query(self, search_type: str, query: str) -> str:
        """Create hash for query"""
        content = f"{search_type}:{query.lower().strip()}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def get(self, search_type: str, query: str) -> Optional[CachedSearchResult]:
        """Get cached result if valid"""
        query_hash = self._hash_query(search_type, query)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT * FROM search_cache WHERE query_hash = ? AND expires_at > ?", (query_hash, time.time())
            )
            row = cursor.fetchone()

        if row:
            return CachedSearchResult(
                query_hash=row[0],
                search_type=row[1],
                query=row[2],
                results=json.loads(row[3]),
                timestamp=row[4],
                expires_at=row[5],
                source=row[6],
            )
        return None

    def store(
        self, search_type: str, query: str, results: Dict[str, Any], ttl_hours: int = 24, source: str = "search"
    ) -> str:
        """Store search result with TTL"""
        query_hash = self._hash_query(search_type, query)
        timestamp = time.time()
        expires_at = timestamp + (ttl_hours * 3600)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO search_cache 
                (query_hash, search_type, original_query, results, timestamp, expires_at, source)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (query_hash, search_type, query, json.dumps(results), timestamp, expires_at, source),
            )

        return query_hash

    def cleanup_expired(self):
        """Remove expired entries"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM search_cache WHERE expires_at < ?", (time.time(),))


class AggressiveQueryAnalyzer:
    """Analyzer that searches for most queries, uses caching"""

    def __init__(self, config_manager):
        self.config_manager = config_manager
        self.cache = SearchCache(config_manager)

        # Categories that should always search
        self.always_search_patterns = [
            r"\b(latest|recent|current|new|update|2024|2025)\b",
            r"\b(how to|tutorial|guide|example|implement)\b",
            r"\b(best|comparison|vs|versus|recommend)\b",
            r"\b(news|release|announcement)\b",
            r"\b(documentation|docs|reference)\b",
        ]

        # Categories that rarely need search
        self.rarely_search_patterns = [
            r"\b(hello|hi|thanks|thank you)\b",
            r"\b(what is|define|definition)\b" + r"(?!.*(latest|current|new))",
            r"\b(math|calculate|solve)\b",
        ]

    async def analyze_query(self, user_query: str, conversation_context: Optional[List[Dict]] = None) -> QueryAnalysisResult:
        """Aggressively determine search needs"""

        import re

        query_lower = user_query.lower()

        # Check for rarely search patterns first
        rarely_search = any(re.search(pattern, query_lower, re.IGNORECASE) for pattern in self.rarely_search_patterns)

        if rarely_search and len(user_query.split()) < 8:
            return QueryAnalysisResult(
                original_query=user_query,
                needs_search=False,
                search_actions=[],
                reasoning="Simple query that doesn't benefit from search",
                confidence=0.9,
            )

        # Check for always search patterns
        always_search = any(re.search(pattern, query_lower, re.IGNORECASE) for pattern in self.always_search_patterns)

        # Default: search for most substantial queries
        should_search = always_search or len(user_query.split()) > 3

        if not should_search:
            return QueryAnalysisResult(
                original_query=user_query,
                needs_search=False,
                search_actions=[],
                reasoning="Short query likely answerable from knowledge",
                confidence=0.7,
            )

        # Determine search types
        actions = []
        max_actions = self.config_manager.get("analysis.max_search_actions", 3)

        # Web search for most queries
        actions.append(SearchAction(action_type=SearchType.WEB_SEARCH, query=user_query, priority=1))

        # Programming queries need both web AND codebase search
        programming_terms = [
            "code",
            "function",
            "class",
            "import",
            "debug",
            "error",
            "programming",
            "python",
            "javascript",
            "react",
            "nodejs",
            "api",
            "library",
            "framework",
            "syntax",
            "example",
        ]

        if any(term in query_lower for term in programming_terms):
            # Add web search for APIs/documentation if not already added
            if len(actions) == 0 or actions[0].action_type != SearchType.WEB_SEARCH:
                actions.insert(
                    0,
                    SearchAction(
                        action_type=SearchType.WEB_SEARCH, query=f"{user_query} API documentation examples", priority=1
                    ),
                )

            # Add codebase search if within max limit
            if len(actions) < max_actions:
                actions.append(
                    SearchAction(action_type=SearchType.CODEBASE_SEARCH, query=f"code examples {user_query}", priority=1)
                )

        # Limit to max_actions
        actions = actions[:max_actions]

        return QueryAnalysisResult(
            original_query=user_query,
            needs_search=True,
            search_actions=actions,
            reasoning=f"Aggressive search strategy - {'always search pattern' if always_search else 'substantial query'}",
            confidence=0.8,
        )


class CachedIterativeChatManager:
    """Chat manager with aggressive search and caching"""

    def __init__(self, config_manager, status_callback=None):
        self.config_manager = config_manager
        self.analyzer = AggressiveQueryAnalyzer(config_manager)
        self.cache = SearchCache(config_manager)
        self.status_callback = status_callback

    async def process_query(self, user_query: str, conversation_context: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """Process query with aggressive search and caching"""

        self._update_status("Analyzing query...")

        # Clean up expired cache entries periodically
        if hash(user_query) % 100 == 0:  # 1% of queries
            self.cache.cleanup_expired()

        # Analyze query
        analysis = await self.analyzer.analyze_query(user_query, conversation_context)

        self._update_status(f"Analysis: {analysis.reasoning}")

        # Execute searches with caching
        search_results = {}
        if analysis.needs_search:
            self._update_status(f"Executing {len(analysis.search_actions)} search actions...")

            for i, action in enumerate(analysis.search_actions, 1):
                # Check cache first
                cached_result = self.cache.get(action.action_type.value, action.query)

                if cached_result:
                    self._update_status(f"Step {i}: Using cached {action.action_type.value}")
                    search_results[f"search_{i}"] = {
                        "action": action,
                        "result": cached_result.results,
                        "cached": True,
                        "cache_age": time.time() - cached_result.timestamp,
                    }
                else:
                    self._update_status(f"Step {i}: {action.action_type.value} - {action.query}")

                    # Execute search
                    result = await self._execute_search_action(action)

                    # Cache successful results
                    if result.get("status") not in ["error", "not_implemented"]:
                        ttl = self._get_cache_ttl(action.action_type.value)
                        self.cache.store(action.action_type.value, action.query, result, ttl)

                    search_results[f"search_{i}"] = {"action": action, "result": result, "cached": False}

        # Generate response
        self._update_status("Generating response...")
        final_response = await self._generate_final_response(user_query, analysis, search_results)

        self._update_status("Complete")

        return {
            "analysis": analysis,
            "search_results": search_results,
            "response": final_response,
            "steps_performed": self._format_steps(analysis, search_results),
        }

    def _get_cache_ttl(self, search_type: str) -> int:
        """Get cache TTL hours based on search type"""
        ttl_map = {
            "WEB_SEARCH": 6,  # 6 hours for web content
            "CODEBASE_SEARCH": 24,  # 24 hours for code
            "FILE_ANALYSIS": 72,  # 72 hours for file analysis
        }
        return ttl_map.get(search_type, 12)

    async def _execute_search_action(self, action: SearchAction) -> Dict[str, Any]:
        """Execute search action - placeholder for now"""
        if action.action_type == SearchType.WEB_SEARCH:
            return {"type": "web", "status": "not_implemented", "query": action.query}
        elif action.action_type == SearchType.CODEBASE_SEARCH:
            return {"type": "codebase", "status": "not_implemented", "query": action.query}
        elif action.action_type == SearchType.FILE_ANALYSIS:
            return {"type": "file", "status": "not_implemented", "query": action.query}
        else:
            return {"type": "unknown", "status": "skipped"}

    async def _generate_final_response(self, user_query: str, analysis, search_results: Dict) -> str:
        """Generate response incorporating search results and cache info"""

        chat_model = self.config_manager.get("ollama.models.chat", "")
        if not chat_model:
            return "No chat model configured"

        # Build context including cache information
        context = ""
        if search_results:
            context += "Available Information:\n"
            for key, result in search_results.items():
                action = result["action"]
                result_data = result["result"]

                if result.get("cached"):
                    age_mins = int(result.get("cache_age", 0) / 60)
                    context += f"- {action.action_type.value} (cached, {age_mins}m old): {action.query}\n"
                else:
                    context += f"- {action.action_type.value} (fresh): {action.query}\n"

                context += f"  Status: {result_data.get('status', 'unknown')}\n"

        prompt = f"""Based on the user's query and available search context, provide a comprehensive response.

User Query: {user_query}

Analysis: {analysis.reasoning}

{context}

Provide a thorough answer that incorporates any relevant information from the searches."""

        try:
            # Use the existing QueryAnalyzer's _call_ollama method
            from query_analyzer import QueryAnalyzer

            analyzer = QueryAnalyzer(self.config_manager)
            response = await analyzer._call_ollama(chat_model, prompt)
            return response
        except Exception as e:
            return f"Error generating response: {str(e)}"

    def _update_status(self, status: str):
        """Update status via callback"""
        if self.status_callback:
            self.status_callback(status)

    def _format_steps(self, analysis, search_results: Dict) -> str:
        """Format steps with cache information"""
        steps = [f"1. Query Analysis: {analysis.reasoning}"]

        if analysis.needs_search:
            cached_count = sum(1 for r in search_results.values() if r.get("cached"))
            fresh_count = len(search_results) - cached_count

            steps.append(f"2. Search Actions: {cached_count} cached, {fresh_count} fresh")

            for i, (key, result) in enumerate(search_results.items(), 1):
                action = result["action"]
                status = result["result"].get("status", "unknown")
                cache_info = " (cached)" if result.get("cached") else " (fresh)"
                steps.append(f"   {i}. {action.action_type.value}: {status}{cache_info}")

        steps.append("3. Generated response with context")
        return "\n".join(steps)