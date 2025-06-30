#!/usr/bin/env python3
"""
Quick test of both research methods with a simple query
"""

import asyncio
import os
from research_interface import ResearchInterface, ResearchMethod

async def quick_test():
    """Run a quick test of both research methods"""
    
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Please set your OPENAI_API_KEY environment variable")
        return
    
    # Simple, focused query for faster testing
    test_query = "What are the top 3 open-source Python frameworks for building LLM applications?"
    
    print("🧪 QUICK TEST: Comparing Both Methods")
    print("="*60)
    print(f"Query: {test_query}")
    print()
    
    # Initialize interface
    interface = ResearchInterface()
    
    try:
        # Test 1: Multi-Agent System (should be faster)
        print("🤖 Testing Multi-Agent System...")
        start_time = asyncio.get_event_loop().time()
        
        multi_result = await interface.research(
            test_query, 
            method=ResearchMethod.MULTI_AGENT,
            verbose=False
        )
        
        multi_time = asyncio.get_event_loop().time() - start_time
        
        print(f"✅ Multi-Agent completed in {multi_time:.1f}s")
        print(f"Result length: {len(multi_result.result)} chars")
        print(f"Metadata: {multi_result.metadata}")
        print(f"Preview: {multi_result.result[:200]}...")
        print()
        
        # Test 2: Deep Research API (will be slower but more comprehensive)
        print("🔬 Testing Deep Research API...")
        print("(This will take longer for comprehensive research...)")
        start_time = asyncio.get_event_loop().time()
        
        deep_result = await interface.research(
            test_query,
            method=ResearchMethod.DEEP_RESEARCH_API,
            verbose=False
        )
        
        deep_time = asyncio.get_event_loop().time() - start_time
        
        print(f"✅ Deep Research completed in {deep_time:.1f}s")
        print(f"Result length: {len(deep_result.result)} chars")
        print(f"Citations: {deep_result.metadata.get('citations_count', 0)}")
        print(f"Preview: {deep_result.result[:200]}...")
        print()
        
        # Comparison
        print("📊 COMPARISON SUMMARY")
        print("-" * 40)
        print(f"Multi-Agent: {multi_time:.1f}s, {len(multi_result.result)} chars")
        print(f"Deep Research: {deep_time:.1f}s, {len(deep_result.result)} chars, {deep_result.metadata.get('citations_count', 0)} citations")
        print()
        
        # Test 3: Auto-selection
        print("🎯 Testing Auto-Selection...")
        auto_result = await interface.research(test_query, verbose=False)
        print(f"✅ Auto-selected: {auto_result.method_used}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    asyncio.run(quick_test())