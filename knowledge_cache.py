import hashlib
import json
import sqlite3
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from query_analyzer import EnhancedLLMQueryAnalyzer, SearchAction, SearchType


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


class CachedIterativeChatManager:
    """Chat manager with aggressive search and caching"""

    def __init__(self, config_manager, status_callback=None, log_callback=None):
        self.config_manager = config_manager
        # Pass log_callback to analyzer
        self.analyzer = EnhancedLLMQueryAnalyzer(config_manager, log_callback=log_callback)
        self.cache = SearchCache(config_manager)
        self.status_callback = status_callback
        self.log_callback = log_callback

    def _log(self, log_type: str, message: str, details: str = ""):
        """Log message via callback if available"""
        if self.log_callback:
            self.log_callback(log_type, message, details)

    async def process_query(self, user_query: str, conversation_context: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """Process query with aggressive search and caching"""

        self._update_status("Analyzing query...")

        # Clean up expired cache entries periodically
        if hash(user_query) % 100 == 0:  # 1% of queries
            self.cache.cleanup_expired()

        # Analyze query
        self._log("INFO", "Starting query analysis", f"Query: {user_query[:100]}...")
        analysis = await self.analyzer.analyze_query(user_query, conversation_context)
        self._log(
            "ANALYSIS", "Query analysis completed", f"Type: {analysis.query_type}, Confidence: {analysis.confidence:.2f}"
        )

        self._update_status(f"Analysis: {analysis.reasoning}")

        # Execute searches with caching
        search_results = {}
        if analysis.needs_search:
            self._update_status(f"Executing {len(analysis.search_actions)} search actions...")
            self._log("INFO", f"Executing {len(analysis.search_actions)} search actions")

            for i, action in enumerate(analysis.search_actions, 1):
                # Check cache first
                cached_result = self.cache.get(action.action_type.value, action.query)

                if cached_result:
                    self._update_status(f"Step {i}: Using cached {action.action_type.value}")
                    self._log("INFO", f"Using cached result for {action.action_type.value}", f"Query: {action.query}")
                    search_results[f"search_{i}"] = {
                        "action": action,
                        "result": cached_result.results,
                        "cached": True,
                        "cache_age": time.time() - cached_result.timestamp,
                    }
                else:
                    self._update_status(f"Step {i}: {action.action_type.value} - {action.query}")
                    self._log("INFO", f"Executing fresh {action.action_type.value}", f"Query: {action.query}")

                    # Execute search
                    result = await self._execute_search_action(action)

                    # Cache successful results
                    if result.get("status") not in ["error", "not_implemented"]:
                        ttl = self._get_cache_ttl(action.action_type.value)
                        self.cache.store(action.action_type.value, action.query, result, ttl)
                        self._log("INFO", f"Cached result for {action.action_type.value}", f"TTL: {ttl}h")

                    search_results[f"search_{i}"] = {"action": action, "result": result, "cached": False}

        # Generate response
        self._update_status("Generating response...")
        final_response = await self._generate_final_response(user_query, analysis, search_results)

        self._update_status("Complete")
        self._log("SUCCESS", "Query processing completed", f"Response length: {len(final_response)} chars")

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
            analyzer = EnhancedLLMQueryAnalyzer(self.config_manager)
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