#!/usr/bin/env python3
"""
Real OpenAI Deep Research Implementation
Implements actual agent-based research workflows using OpenAI's agents API
"""

import asyncio
import json
import os
from typing import List, Dict, Optional
from pydantic import BaseModel

# Import OpenAI agents
from agents import Agent, Runner, RunConfig, WebSearchTool, HostedMCPTool

# Environment setup
os.environ.setdefault("OPENAI_AGENTS_DISABLE_TRACING", "1")

# Pydantic models for structured output
class Clarifications(BaseModel):
    questions: List[str]

class ResearchResult(BaseModel):
    summary: str
    key_findings: List[str]
    sources: List[str]
    recommendations: List[str]

# Agent prompts
TRIAGE_AGENT_PROMPT = """
You are a research triage agent. Your job is to analyze incoming research queries and decide whether clarifying questions are needed before proceeding with research.

If the query is:
- Clear and specific: Transfer directly to the Research Instruction Agent
- Vague or ambiguous: Transfer to the Clarifying Questions Agent

Consider factors like:
- Topic specificity
- Scope clarity
- Time frame definition
- Target audience clarity
"""

CLARIFYING_AGENT_PROMPT = """
You are a clarifying questions agent. When you receive a research query that needs more context, ask 2-3 targeted clarifying questions to gather essential information before research begins.

Focus on:
- Scope and boundaries
- Target audience or use case
- Time frame or currency requirements
- Specific aspects of interest
- Depth vs breadth preferences

After receiving answers, transfer to the Research Instruction Agent.
"""

RESEARCH_INSTRUCTION_AGENT_PROMPT = """
You are a research instruction agent. Your job is to take user queries (and any clarifying information) and transform them into detailed, structured research instructions for the Research Agent.

Create comprehensive research instructions that include:
- Clear research objectives
- Key areas to investigate
- Types of sources to prioritize
- Specific questions to answer
- Expected deliverables format

Then transfer to the Research Agent with these detailed instructions.
"""

RESEARCH_AGENT_PROMPT = """
You are a deep research agent specializing in comprehensive, empirical research. You perform thorough investigations using web search and other available tools to gather authoritative information.

Your research approach:
1. Break down complex topics into key research areas
2. Search for authoritative sources (academic papers, industry reports, official documentation)
3. Gather diverse perspectives and current information
4. Synthesize findings into coherent insights
5. Provide specific evidence and citations
6. Identify gaps or limitations in available information

Always provide:
- Executive summary
- Key findings with supporting evidence
- Source citations with URLs
- Actionable recommendations where appropriate
- Areas for further investigation
"""


class DeepResearchSystem:
    """Real implementation of the Deep Research system using OpenAI agents"""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the research system with OpenAI API key"""
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        elif not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        self.setup_agents()
    
    def setup_agents(self):
        """Set up the multi-agent research pipeline"""
        
        # Research Agent - the core research performer
        self.research_agent = Agent(
            name="Research Agent",
            model="gpt-4o",  # Using gpt-4o as the deep research model may not be available
            instructions=RESEARCH_AGENT_PROMPT,
            tools=[
                WebSearchTool(),
                # Add MCP tools if needed
                # HostedMCPTool({
                #     "type": "mcp",
                #     "server_label": "file_search",
                #     "server_url": "https://api.example.com/mcp",
                #     "require_approval": "never",
                # })
            ]
        )
        
        # Research Instruction Agent
        self.instruction_agent = Agent(
            name="Research Instruction Agent",
            model="gpt-4o-mini",
            instructions=RESEARCH_INSTRUCTION_AGENT_PROMPT,
            handoffs=[self.research_agent]
        )
        
        # Clarifying Questions Agent
        self.clarifying_agent = Agent(
            name="Clarifying Questions Agent",
            model="gpt-4o-mini",
            instructions=CLARIFYING_AGENT_PROMPT,
            output_type=Clarifications,
            handoffs=[self.instruction_agent]
        )
        
        # Triage Agent - entry point
        self.triage_agent = Agent(
            name="Triage Agent",
            model="gpt-4o-mini",
            instructions=TRIAGE_AGENT_PROMPT,
            handoffs=[self.clarifying_agent, self.instruction_agent]
        )
    
    async def basic_research(self, query: str, verbose: bool = True) -> str:
        """Perform basic research using just the research agent"""
        if verbose:
            print(f"🔍 Starting research: {query}\n")
        
        result_stream = Runner.run_streamed(
            starting_agent=self.research_agent,
            input=query,
            run_config=RunConfig(tracing_disabled=True)
        )
        
        if verbose:
            await self._stream_events(result_stream)
        
        # Get final output (stream events until completion)
        final_output = None
        async for event in result_stream.stream_events():
            pass  # Just consume events to completion
        
        # Try to get the final output
        try:
            final_output = result_stream.final_output_as(str)
        except Exception as e:
            final_output = f"Research completed but could not retrieve final output: {e}"
        
        return final_output
    
    async def multi_agent_research(
        self, 
        query: str, 
        clarification_answers: Optional[Dict[str, str]] = None,
        verbose: bool = True
    ) -> str:
        """Perform research using the full multi-agent pipeline"""
        if verbose:
            print(f"🤖 Starting multi-agent research: {query}\n")
        
        result_stream = Runner.run_streamed(
            starting_agent=self.triage_agent,
            input=query,
            run_config=RunConfig(tracing_disabled=True)
        )
        
        if verbose:
            await self._stream_events(result_stream)
        
        # Get final output (stream events until completion)
        final_output = None
        async for event in result_stream.stream_events():
            pass  # Just consume events to completion
        
        # Try to get the final output
        try:
            final_output = result_stream.final_output_as(str)
        except Exception as e:
            final_output = f"Research completed but could not retrieve final output: {e}"
        
        if verbose:
            print("\n" + "="*80)
            print("📋 AGENT INTERACTION FLOW")
            print("="*80)
            print("Multi-agent workflow completed")
            
        return final_output
    
    async def _stream_events(self, result_stream):
        """Stream and display events from the research process"""
        try:
            async for event in result_stream.stream_events():
                if hasattr(event, 'type'):
                    if event.type == "agent_updated_stream_event":
                        if hasattr(event, 'new_agent'):
                            print(f"\n🔄 Switched to agent: {event.new_agent.name}")
                    elif event.type == "run_item_stream_event":
                        if hasattr(event, 'item') and hasattr(event.item, 'type'):
                            if event.item.type == "tool_call_item":
                                tool_name = getattr(event.item.raw_item, 'name', 'Unknown tool')
                                if 'search' in tool_name.lower():
                                    print(f"🔍 [Web Search] {tool_name}")
                else:
                    # Handle other event types
                    print(f"📡 Event: {type(event).__name__}")
        except Exception as e:
            print(f"⚠️  Streaming error: {e}")
            # Continue without streaming display
    
    def _print_agent_flow(self, result_stream):
        """Print the agent interaction flow"""
        interactions = []
        count = 1
        
        for item in result_stream.new_items:
            agent_name = getattr(item.agent, "name", "Unknown Agent") if hasattr(item, "agent") else "Unknown Agent"
            
            if item.type == "handoff_call_item":
                func_name = getattr(item.raw_item, "name", "Unknown Function")
                interactions.append(f"{count}. [{agent_name}] → Handoff: {func_name}")
                count += 1
            elif item.type == "tool_call_item":
                tool_name = getattr(item.raw_item, "name", None)
                if tool_name and tool_name.strip():
                    args = getattr(item.raw_item, "arguments", None)
                    args_str = ""
                    if args:
                        try:
                            parsed_args = json.loads(args)
                            if parsed_args:
                                args_str = json.dumps(parsed_args)
                        except Exception:
                            if args.strip() and args.strip() != "{}":
                                args_str = args.strip()
                    
                    args_display = f" with args {args_str}" if args_str else ""
                    interactions.append(f"{count}. [{agent_name}] → Tool: {tool_name}{args_display}")
                    count += 1
            elif item.type == "message_output_item":
                interactions.append(f"{count}. [{agent_name}] → Output")
                count += 1
        
        for interaction in interactions:
            print(interaction)
    
    def _print_citations(self, result_stream, preceding_chars: int = 50):
        """Extract and print citations from the research output"""
        citations = []
        
        for item in reversed(result_stream.new_items):
            if item.type == "message_output_item":
                for content in getattr(item.raw_item, 'content', []):
                    if not hasattr(content, 'annotations') or not hasattr(content, 'text'):
                        continue
                    
                    text = content.text
                    for ann in content.annotations:
                        if getattr(ann, 'type', None) == 'url_citation':
                            title = getattr(ann, 'title', '<no title>')
                            url = getattr(ann, 'url', '<no url>')
                            start = getattr(ann, 'start_index', None)
                            end = getattr(ann, 'end_index', None)
                            
                            if start is not None and end is not None and isinstance(text, str):
                                pre_start = max(0, start - preceding_chars)
                                preceding_text = text[pre_start:start].replace('\n', ' ').strip()
                                excerpt = text[start:end].replace('\n', ' ').strip()
                                
                                citations.append({
                                    'title': title,
                                    'url': url,
                                    'preceding_text': preceding_text,
                                    'excerpt': excerpt
                                })
                break
        
        if citations:
            for i, citation in enumerate(citations, 1):
                print(f"{i}. {citation['title']}")
                print(f"   URL: {citation['url']}")
                print(f"   Context: ...{citation['preceding_text']} [{citation['excerpt']}]")
                print()
        else:
            print("No citations found in the research output.")


def print_usage():
    """Print usage information and available parameters"""
    print("🚀 OpenAI Multi-Agent Research System")
    print("="*50)
    print("Interactive research using custom multi-agent orchestration")
    print()
    print("📋 Available Parameters:")
    print("  --query, -q         Research question/topic (interactive if not provided)")
    print("  --mode, -m          Research mode:")
    print("                      • basic (default, direct to research agent)")
    print("                      • multi-agent (full pipeline with triage)")
    print("  --clarify, -c       Provide clarification answers (key=value pairs)")
    print("  --verbose, -v       Show detailed progress information")
    print("  --help, -h          Show this help message")
    print()
    print("🎯 Usage Examples:")
    print("  python multi_agent_research.py")
    print("  python multi_agent_research.py -q 'LLM frameworks comparison'")
    print("  python multi_agent_research.py -m multi-agent -q 'AI trends'")
    print("  python multi_agent_research.py --verbose")
    print()

def get_interactive_input():
    """Get research parameters interactively"""
    print("🔍 Interactive Multi-Agent Research Setup")
    print("-" * 45)
    
    # Get research query
    print("\n📝 Enter your research question:")
    query = input("Query: ").strip()
    if not query:
        print("❌ Query cannot be empty")
        return None
    
    # Get research mode
    print("\n🤖 Select research mode:")
    print("1. basic (direct to research agent, faster)")
    print("2. multi-agent (full pipeline with triage, more thorough)")
    mode_choice = input("Choice (1 or 2, default=1): ").strip()
    
    if mode_choice == "2":
        mode = "multi-agent"
        
        # Get clarifications for multi-agent mode
        print("\n💬 Clarification answers (optional):")
        print("Press Enter for each prompt to skip, or provide answers to guide research")
        
        clarifications = {}
        common_questions = [
            "What specific aspects are most important?",
            "What timeframe should be considered?",
            "What is the target audience or use case?",
            "What level of detail is needed?"
        ]
        
        for question in common_questions:
            answer = input(f"{question}: ").strip()
            if answer:
                clarifications[question] = answer
    else:
        mode = "basic"
        clarifications = {}
    
    # Get verbosity preference
    print("\n📊 Show detailed progress information?")
    verbose_choice = input("Verbose mode (y/n, default=y): ").strip().lower()
    verbose = verbose_choice != "n"
    
    return {
        "query": query,
        "mode": mode,
        "clarifications": clarifications,
        "verbose": verbose
    }

# Example usage and testing functions
async def main():
    """Main function with interactive parameter selection"""
    import sys
    
    # Parse command line arguments
    args = {}
    clarifications = {}
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        
        if arg in ["-h", "--help"]:
            print_usage()
            return
        elif arg in ["-q", "--query"] and i + 1 < len(sys.argv):
            args["query"] = sys.argv[i + 1]
            i += 2
        elif arg in ["-m", "--mode"] and i + 1 < len(sys.argv):
            args["mode"] = sys.argv[i + 1]
            i += 2
        elif arg in ["-c", "--clarify"] and i + 1 < len(sys.argv):
            # Parse key=value pairs
            clarify_pair = sys.argv[i + 1]
            if "=" in clarify_pair:
                key, value = clarify_pair.split("=", 1)
                clarifications[key] = value
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
        if not clarifications:
            clarifications = interactive_params.get("clarifications", {})
    
    # Set defaults
    query = args["query"]
    mode = args.get("mode", "basic")
    verbose = args.get("verbose", True)
    
    print(f"\n🚀 Starting Multi-Agent Research")
    print("="*50)
    print(f"Mode: {mode}")
    print(f"Query: {query}")
    print(f"Verbose: {verbose}")
    if clarifications:
        print(f"Clarifications: {len(clarifications)} provided")
    print()
    
    # Initialize the research system
    try:
        research_system = DeepResearchSystem()
    except Exception as e:
        print(f"❌ Failed to initialize research system: {e}")
        return
    
    try:
        if mode == "multi-agent":
            print("🤖 MULTI-AGENT RESEARCH MODE")
            print("-" * 40)
            
            result = await research_system.multi_agent_research(
                query, 
                clarification_answers=clarifications if clarifications else None,
                verbose=verbose
            )
            
        else:  # basic mode
            print("🔍 BASIC RESEARCH MODE")
            print("-" * 40)
            
            result = await research_system.basic_research(query, verbose=verbose)
        
        print(f"\n📄 Research Result:")
        print("="*50)
        print(result)
        print("\n✅ Research completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during research: {e}")
        print("Make sure your OpenAI API key is valid and you have sufficient credits.")


if __name__ == "__main__":
    asyncio.run(main())