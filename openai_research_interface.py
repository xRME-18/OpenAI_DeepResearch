#!/usr/bin/env python3
"""
Unified Research Interface
Provides a single interface to use either Multi-Agent or Deep Research API approaches
"""

import asyncio
import os
from enum import Enum
from typing import Optional, Dict, Any, Union
from pydantic import BaseModel

# Import both OpenAI implementations
from openai_agents_research import DeepResearchSystem as OpenAIAgentsSystem
from openai_deep_research_api import OpenAIDeepResearchAPI, DeepResearchResult


class ResearchMethod(Enum):
    """Available OpenAI research methods"""
    OPENAI_AGENTS = "openai_agents"
    DEEP_RESEARCH_API = "deep_research_api"
    AUTO = "auto"  # Automatically choose best method


class UnifiedResearchResult(BaseModel):
    """Unified result format for both research methods"""
    query: str
    method_used: str
    result: str
    metadata: Dict[str, Any]


class ResearchInterface:
    """
    Unified interface for OpenAI research approaches
    
    Usage:
        # Auto-select best method
        interface = ResearchInterface()
        result = await interface.research("What are LLM frameworks?")
        
        # Force specific method
        interface = ResearchInterface(method=ResearchMethod.OPENAI_AGENTS)
        result = await interface.research("What are LLM frameworks?")
    """
    
    def __init__(
        self, 
        method: ResearchMethod = ResearchMethod.AUTO,
        api_key: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the research interface
        
        Args:
            method: Which research method to use
            api_key: OpenAI API key
            **kwargs: Additional configuration for specific methods
        """
        self.method = method
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.config = kwargs
        
        # Initialize OpenAI systems based on method
        self.openai_agents_system = None
        self.deep_research_api = None
        
        if method in [ResearchMethod.OPENAI_AGENTS, ResearchMethod.AUTO]:
            try:
                self.openai_agents_system = OpenAIAgentsSystem(api_key=self.api_key)
                print("✅ OpenAI Agents system initialized")
            except Exception as e:
                print(f"⚠️  OpenAI Agents system failed to initialize: {e}")
        
        if method in [ResearchMethod.DEEP_RESEARCH_API, ResearchMethod.AUTO]:
            try:
                model = kwargs.get("deep_research_model", "o3-deep-research-2025-06-26")
                self.deep_research_api = OpenAIDeepResearchAPI(
                    api_key=self.api_key, 
                    model=model
                )
                print("✅ OpenAI Deep Research API initialized")
            except Exception as e:
                print(f"⚠️  OpenAI Deep Research API failed to initialize: {e}")
    
    async def research(
        self, 
        query: str, 
        method: Optional[ResearchMethod] = None,
        verbose: bool = True,
        **kwargs
    ) -> UnifiedResearchResult:
        """
        Perform research using the specified or auto-selected method
        
        Args:
            query: Research question or topic
            method: Override the default method for this query
            verbose: Show progress information
            **kwargs: Method-specific parameters
        
        Returns:
            UnifiedResearchResult with findings and metadata
        """
        # Determine which method to use
        effective_method = method or self.method
        
        if effective_method == ResearchMethod.AUTO:
            effective_method = self._auto_select_method(query)
        
        if verbose:
            print(f"🔍 Using method: {effective_method.value}")
        
        # Execute research based on method
        if effective_method == ResearchMethod.OPENAI_AGENTS:
            return await self._research_openai_agents(query, verbose, **kwargs)
        elif effective_method == ResearchMethod.DEEP_RESEARCH_API:
            return await self._research_deep_api(query, verbose, **kwargs)
        else:
            raise ValueError(f"Unknown research method: {effective_method}")
    
    async def _research_openai_agents(
        self, 
        query: str, 
        verbose: bool,
        **kwargs
    ) -> UnifiedResearchResult:
        """Research using OpenAI Agents system"""
        if not self.openai_agents_system:
            raise RuntimeError("OpenAI Agents system not available")
        
        # Use OpenAI Agents research
        clarifications = kwargs.get("clarifications", {})
        
        if clarifications:
            result = await self.openai_agents_system.multi_agent_research(
                query, 
                clarification_answers=clarifications,
                verbose=verbose
            )
        else:
            result = await self.openai_agents_system.basic_research(query, verbose=verbose)
        
        return UnifiedResearchResult(
            query=query,
            method_used="openai_agents",
            result=result,
            metadata={
                "agents_used": ["triage", "clarifying", "instruction", "research"],
                "has_clarifications": bool(clarifications),
                "approach": "openai_agents_orchestration"
            }
        )
    
    async def _research_deep_api(
        self, 
        query: str, 
        verbose: bool,
        **kwargs
    ) -> UnifiedResearchResult:
        """Research using Deep Research API"""
        if not self.deep_research_api:
            raise RuntimeError("Deep Research API not available")
        
        # Use Deep Research API
        deep_result = await self.deep_research_api.research(query, **kwargs)
        
        return UnifiedResearchResult(
            query=query,
            method_used="deep_research_api",
            result=deep_result.text,
            metadata={
                "model": self.deep_research_api.model,
                "citations_count": len(deep_result.citations),
                "reasoning_steps_count": len(deep_result.reasoning_steps),
                "web_searches_count": len(deep_result.web_searches),
                "approach": "native_deep_research",
                "citations": [
                    {"title": c.title, "url": c.url, "excerpt": c.excerpt[:100]}
                    for c in deep_result.citations[:5]  # First 5 citations for metadata
                ]
            }
        )
    
    def _auto_select_method(self, query: str) -> ResearchMethod:
        """
        Automatically select the best research method based on query characteristics
        
        Args:
            query: Research question to analyze
            
        Returns:
            Best research method for this query
        """
        # Simple heuristics for method selection
        query_lower = query.lower()
        
        # Prefer Deep Research API for:
        # - Complex, open-ended research questions
        # - Queries requiring comprehensive analysis
        if any(keyword in query_lower for keyword in [
            "landscape", "comprehensive", "current state", "overview", 
            "compare", "analysis", "trends", "future"
        ]):
            if self.deep_research_api:
                return ResearchMethod.DEEP_RESEARCH_API
        
        # Prefer OpenAI Agents for:
        # - Specific technical questions
        # - Queries that might need clarification
        if any(keyword in query_lower for keyword in [
            "how to", "implement", "specific", "technical", "which", "best"
        ]):
            if self.openai_agents_system:
                return ResearchMethod.OPENAI_AGENTS
        
        # Default: use whatever is available
        if self.deep_research_api:
            return ResearchMethod.DEEP_RESEARCH_API
        elif self.openai_agents_system:
            return ResearchMethod.OPENAI_AGENTS
        else:
            raise RuntimeError("No research methods available")
    
    def research_sync(self, query: str, **kwargs) -> UnifiedResearchResult:
        """Synchronous version of research method"""
        return asyncio.run(self.research(query, **kwargs))


def print_usage():
    """Print usage information and available parameters"""
    print("🚀 OpenAI Research Interface")
    print("="*50)
    print("Intelligent routing between OpenAI Agents and Deep Research API approaches")
    print()
    print("📋 Available Parameters:")
    print("  --query, -q         Research question/topic (interactive if not provided)")
    print("  --method, -m        Research method:")
    print("                      • auto (default, intelligent selection)")
    print("                      • openai-agents (custom orchestration)")
    print("                      • deep-research (native OpenAI API)")
    print("  --model             Deep research model (o3/o4-mini-deep-research)")
    print("  --system, -s        Custom system message for research guidance")
    print("  --verbose, -v       Show detailed progress information")
    print("  --help, -h          Show this help message")
    print()
    print("🎯 Usage Examples:")
    print("  python openai_research_interface.py")
    print("  python openai_research_interface.py -q 'LLM frameworks comparison'")
    print("  python openai_research_interface.py -m deep-research -q 'AI trends'")
    print("  python openai_research_interface.py --model o4-mini-deep-research-2025-06-26")
    print()

def get_interactive_input():
    """Get research parameters interactively"""
    print("🔍 Interactive OpenAI Research Setup")
    print("-" * 40)
    
    # Get research query
    print("\n📝 Enter your research question:")
    query = input("Query: ").strip()
    if not query:
        print("❌ Query cannot be empty")
        return None
    
    # Get method preference
    print("\n🤖 Select OpenAI research method:")
    print("1. auto (intelligent selection based on query)")
    print("2. openai-agents (custom orchestration, faster)")
    print("3. deep-research (native OpenAI API, comprehensive)")
    method_choice = input("Choice (1-3, default=1): ").strip()
    
    method_map = {
        "1": ResearchMethod.AUTO,
        "2": ResearchMethod.OPENAI_AGENTS,
        "3": ResearchMethod.DEEP_RESEARCH_API
    }
    method = method_map.get(method_choice, ResearchMethod.AUTO)
    
    # Get model preference for deep research
    model = None
    if method in [ResearchMethod.DEEP_RESEARCH_API, ResearchMethod.AUTO]:
        print("\n🧠 Select Deep Research model (if used):")
        print("1. o3-deep-research-2025-06-26 (most capable, slower)")
        print("2. o4-mini-deep-research-2025-06-26 (faster, efficient)")
        model_choice = input("Choice (1 or 2, default=1): ").strip()
        
        if model_choice == "2":
            model = "o4-mini-deep-research-2025-06-26"
        else:
            model = "o3-deep-research-2025-06-26"
    
    # Get optional system message
    print("\n💬 Custom system message (optional):")
    print("(Press Enter to use default research prompts)")
    system_message = input("System message: ").strip()
    if not system_message:
        system_message = None
    
    # Get verbosity preference
    print("\n📊 Show detailed progress information?")
    verbose_choice = input("Verbose mode (y/n, default=y): ").strip().lower()
    verbose = verbose_choice != "n"
    
    return {
        "query": query,
        "method": method,
        "model": model,
        "system_message": system_message,
        "verbose": verbose
    }

async def main():
    """Main function with interactive parameter selection"""
    import sys
    
    # Parse command line arguments
    args = {}
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        
        if arg in ["-h", "--help"]:
            print_usage()
            return
        elif arg in ["-q", "--query"] and i + 1 < len(sys.argv):
            args["query"] = sys.argv[i + 1]
            i += 2
        elif arg in ["-m", "--method"] and i + 1 < len(sys.argv):
            method_str = sys.argv[i + 1]
            method_map = {
                "auto": ResearchMethod.AUTO,
                "openai-agents": ResearchMethod.OPENAI_AGENTS,
                "deep-research": ResearchMethod.DEEP_RESEARCH_API
            }
            args["method"] = method_map.get(method_str, ResearchMethod.AUTO)
            i += 2
        elif arg == "--model" and i + 1 < len(sys.argv):
            args["model"] = sys.argv[i + 1]
            i += 2
        elif arg in ["-s", "--system"] and i + 1 < len(sys.argv):
            args["system_message"] = sys.argv[i + 1]
            i += 2
        elif arg in ["-v", "--verbose"]:
            args["verbose"] = True
            i += 1
        else:
            print(f"❌ Unknown argument: {arg}")
            print("Use --help for usage information")
            return
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Please set your OPENAI_API_KEY environment variable")
        print("   export OPENAI_API_KEY='your-api-key-here'")
        return
    
    # Get parameters interactively if not provided via command line
    if not args.get("query"):
        interactive_params = get_interactive_input()
        if not interactive_params:
            return
        args.update(interactive_params)
    
    # Set defaults
    query = args["query"]
    method = args.get("method", ResearchMethod.AUTO)
    model = args.get("model", "o3-deep-research-2025-06-26")
    system_message = args.get("system_message")
    verbose = args.get("verbose", True)
    
    print(f"\n🚀 Starting OpenAI Research")
    print("="*50)
    print(f"Method: {method.value}")
    print(f"Query: {query}")
    if method in [ResearchMethod.DEEP_RESEARCH_API, ResearchMethod.AUTO]:
        print(f"Deep Research Model: {model}")
    if system_message:
        print(f"System: {system_message[:50]}...")
    print(f"Verbose: {verbose}")
    print()
    
    # Initialize interface
    try:
        interface_kwargs = {}
        if model:
            interface_kwargs["deep_research_model"] = model
            
        interface = ResearchInterface(method=method, **interface_kwargs)
    except Exception as e:
        print(f"❌ Failed to initialize research interface: {e}")
        return
    
    # Perform research
    try:
        result = await interface.research(
            query=query,
            verbose=verbose,
            system_message=system_message
        )
        
        print(f"\n📄 Research Result:")
        print("="*50)
        print(f"Method Used: {result.method_used}")
        print(f"Metadata: {result.metadata}")
        print("\n" + "="*50)
        print("RESEARCH FINDINGS:")
        print("="*50)
        print(result.result)
        
        # Show additional metadata for deep research
        if result.method_used == "deep_research_api":
            metadata = result.metadata
            if metadata.get("citations"):
                print(f"\n📚 Citations ({metadata.get('citations_count', 0)}):")
                for i, citation in enumerate(metadata["citations"], 1):
                    print(f"{i}. {citation['title']}")
                    if citation.get('url'):
                        print(f"   URL: {citation['url']}")
                    if citation.get('excerpt'):
                        print(f"   Excerpt: {citation['excerpt']}")
                    print()
        elif result.method_used == "openai_agents":
            print(f"\n🤖 Used OpenAI Agents: {', '.join(result.metadata.get('agents_used', []))}")
        
        print("\n✅ Research completed successfully!")
        
    except Exception as e:
        print(f"❌ Research error: {e}")


if __name__ == "__main__":
    asyncio.run(main())