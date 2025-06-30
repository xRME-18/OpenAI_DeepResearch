#!/usr/bin/env python3
"""
OpenAI Deep Research API Implementation
Uses the actual Deep Research API with o3-deep-research or o4-mini-deep-research models
"""

import asyncio
import os
from typing import Optional, Dict, Any, List
from pydantic import BaseModel
from openai import OpenAI


class Citation(BaseModel):
    """Citation from Deep Research API"""
    title: str
    url: str
    start_index: int
    end_index: int
    excerpt: str

class DeepResearchResult(BaseModel):
    """Result from Deep Research API"""
    text: str
    citations: List[Citation]
    reasoning_steps: List[str]
    web_searches: List[str]
    raw_response: Any


class OpenAIDeepResearchAPI:
    """
    Wrapper for OpenAI's Deep Research API using the responses endpoint
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "o3-deep-research-2025-06-26"):
        """
        Initialize Deep Research API client
        
        Args:
            api_key: OpenAI API key
            model: Deep research model to use:
                   - "o3-deep-research-2025-06-26" (more capable, slower)
                   - "o4-mini-deep-research-2025-06-26" (faster, lightweight)
        """
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        elif not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OpenAI API key required")
        
        self.model = model
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"), timeout=600.0)
        
        print(f"🔬 Initialized Deep Research API with model: {model}")
    
    async def research(self, query: str, system_message: Optional[str] = None, **kwargs) -> DeepResearchResult:
        """
        Perform deep research using OpenAI's Deep Research API
        
        Args:
            query: Research question or topic
            system_message: Optional system message to guide the research
            **kwargs: Additional parameters (summary, tools, etc.)
        
        Returns:
            DeepResearchResult with text, citations, and metadata
        """
        print(f"🔍 Starting Deep Research API query: {query[:100]}...")
        
        # Default system message for professional research
        if not system_message:
            system_message = """
            You are a professional researcher preparing a structured, data-driven report. 
            Focus on data-rich insights with specific figures, trends, and measurable outcomes.
            Prioritize reliable, up-to-date sources and include inline citations.
            Be analytical and ensure each section supports data-backed reasoning.
            """
        
        # Prepare the input messages
        input_messages = []
        
        if system_message:
            input_messages.append({
                "role": "developer",
                "content": [{"type": "input_text", "text": system_message}]
            })
        
        input_messages.append({
            "role": "user",
            "content": [{"type": "input_text", "text": query}]
        })
        
        # Configure tools - always include web search
        tools = kwargs.get("tools", [
            {"type": "web_search_preview"},
            {"type": "code_interpreter", "container": {"type": "auto", "file_ids": []}}
        ])
        
        # Set reasoning summary level (optional for verified orgs)
        reasoning_config = kwargs.get("reasoning", None)
        if reasoning_config is None:
            # Try with reasoning first, fall back without if not verified
            try:
                reasoning_summary = kwargs.get("summary", "auto")
                reasoning_config = {"summary": reasoning_summary}
            except:
                reasoning_config = None
        
        try:
            # Make the actual Deep Research API call
            request_params = {
                "model": self.model,
                "input": input_messages,
                "tools": tools
            }
            
            # Try with reasoning first, fall back without if organization not verified
            response = None
            if reasoning_config:
                try:
                    request_params["reasoning"] = reasoning_config
                    response = self.client.responses.create(**request_params)
                    print("✅ Deep Research with reasoning enabled")
                except Exception as reasoning_error:
                    if "must be verified" in str(reasoning_error):
                        print("⚠️  Organization not verified for reasoning, proceeding without reasoning summaries")
                        # Remove reasoning and try again
                        request_params.pop("reasoning", None)
                        response = self.client.responses.create(**request_params)
                    else:
                        raise reasoning_error
            else:
                response = self.client.responses.create(**request_params)
            
            # Extract the final output text
            final_output = response.output[-1].content[0].text
            
            # Extract citations
            citations = []
            if hasattr(response.output[-1].content[0], 'annotations'):
                for annotation in response.output[-1].content[0].annotations:
                    if hasattr(annotation, 'start_index'):
                        # Extract excerpt from the text
                        start = annotation.start_index
                        end = annotation.end_index
                        excerpt = final_output[start:end] if start < len(final_output) else ""
                        
                        citations.append(Citation(
                            title=getattr(annotation, 'title', 'Unknown Title'),
                            url=getattr(annotation, 'url', ''),
                            start_index=start,
                            end_index=end,
                            excerpt=excerpt
                        ))
            
            # Extract reasoning steps
            reasoning_steps = []
            for item in response.output:
                if item.type == "reasoning" and hasattr(item, 'summary'):
                    for summary_item in item.summary:
                        if hasattr(summary_item, 'text'):
                            reasoning_steps.append(summary_item.text)
            
            # Extract web search queries
            web_searches = []
            for item in response.output:
                if item.type == "web_search_call" and hasattr(item, 'action'):
                    if 'query' in item.action:
                        web_searches.append(item.action['query'])
            
            print(f"✅ Deep Research completed: {len(citations)} citations, {len(reasoning_steps)} reasoning steps")
            
            return DeepResearchResult(
                text=final_output,
                citations=citations,
                reasoning_steps=reasoning_steps,
                web_searches=web_searches,
                raw_response=response
            )
            
        except Exception as e:
            print(f"❌ Deep Research API error: {e}")
            # Return a basic result with error info
            return DeepResearchResult(
                text=f"Error during research: {e}",
                citations=[],
                reasoning_steps=[],
                web_searches=[],
                raw_response=None
            )
    
    def research_sync(self, query: str, **kwargs) -> DeepResearchResult:
        """Synchronous version of research method"""
        return asyncio.run(self.research(query, **kwargs))


def print_usage():
    """Print usage information and available parameters"""
    print("🚀 OpenAI Deep Research API")
    print("="*50)
    print("Interactive Deep Research using OpenAI's specialized research models")
    print()
    print("📋 Available Parameters:")
    print("  --model, -m         Deep research model to use:")
    print("                      • o3-deep-research-2025-06-26 (default, most capable)")
    print("                      • o4-mini-deep-research-2025-06-26 (faster)")
    print("  --query, -q         Research question/topic (interactive if not provided)")
    print("  --system, -s        Custom system message for research guidance")
    print("  --summary           Reasoning summary level (auto, detailed, none)")
    print("  --help, -h          Show this help message")
    print()
    print("🎯 Usage Examples:")
    print("  python deep_research_api.py")
    print("  python deep_research_api.py -q 'LLM frameworks comparison'")
    print("  python deep_research_api.py -m o4-mini-deep-research-2025-06-26 -q 'AI trends'")
    print("  python deep_research_api.py --system 'Focus on technical implementation'")
    print()

def get_interactive_input():
    """Get research parameters interactively"""
    print("🔍 Interactive Deep Research Setup")
    print("-" * 40)
    
    # Get research query
    print("\n📝 Enter your research question:")
    query = input("Query: ").strip()
    if not query:
        print("❌ Query cannot be empty")
        return None
    
    # Get model choice
    print("\n🤖 Select research model:")
    print("1. o3-deep-research-2025-06-26 (most capable, slower)")
    print("2. o4-mini-deep-research-2025-06-26 (faster, efficient)")
    model_choice = input("Choice (1 or 2, default=1): ").strip()
    
    if model_choice == "2":
        model = "o4-mini-deep-research-2025-06-26"
    else:
        model = "o3-deep-research-2025-06-26"
    
    # Get optional system message
    print("\n💬 Custom system message (optional):")
    print("(Press Enter to use default professional research prompt)")
    system_message = input("System message: ").strip()
    if not system_message:
        system_message = None
    
    # Get reasoning summary preference
    print("\n🧠 Reasoning summary level:")
    print("1. auto (default)")
    print("2. detailed")
    print("3. none")
    summary_choice = input("Choice (1-3, default=1): ").strip()
    
    summary_map = {"1": "auto", "2": "detailed", "3": "none"}
    summary = summary_map.get(summary_choice, "auto")
    
    return {
        "query": query,
        "model": model,
        "system_message": system_message,
        "summary": summary
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
        elif arg in ["-m", "--model"] and i + 1 < len(sys.argv):
            args["model"] = sys.argv[i + 1]
            i += 2
        elif arg in ["-s", "--system"] and i + 1 < len(sys.argv):
            args["system_message"] = sys.argv[i + 1]
            i += 2
        elif arg == "--summary" and i + 1 < len(sys.argv):
            args["summary"] = sys.argv[i + 1]
            i += 2
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
    model = args.get("model", "o3-deep-research-2025-06-26")
    query = args["query"]
    system_message = args.get("system_message")
    summary = args.get("summary", "auto")
    
    print(f"\n🚀 Starting Deep Research")
    print("="*50)
    print(f"Model: {model}")
    print(f"Query: {query}")
    print(f"Summary: {summary}")
    if system_message:
        print(f"System: {system_message[:50]}...")
    print()
    
    # Initialize API
    try:
        api = OpenAIDeepResearchAPI(model=model)
    except Exception as e:
        print(f"❌ Failed to initialize API: {e}")
        return
    
    # Perform research
    try:
        result = await api.research(
            query=query,
            system_message=system_message,
            summary=summary
        )
        
        print(f"\n📄 Deep Research Results:")
        print("="*50)
        print(result.text)
        
        if result.web_searches:
            print(f"\n🔍 Web Searches Performed ({len(result.web_searches)}):")
            for i, search in enumerate(result.web_searches, 1):
                print(f"{i}. {search}")
        
        if result.reasoning_steps:
            print(f"\n🧠 Reasoning Steps ({len(result.reasoning_steps)}):")
            for i, step in enumerate(result.reasoning_steps, 1):
                print(f"{i}. {step}")
        
        if result.citations:
            print(f"\n📚 Citations ({len(result.citations)}):")
            for i, citation in enumerate(result.citations, 1):
                print(f"{i}. {citation.title}")
                print(f"   URL: {citation.url}")
                print(f"   Excerpt: {citation.excerpt}")
                print()
        
        print("✅ Research completed successfully!")
            
    except Exception as e:
        print(f"❌ Research error: {e}")


if __name__ == "__main__":
    asyncio.run(main())