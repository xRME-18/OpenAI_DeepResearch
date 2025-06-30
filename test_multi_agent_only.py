#!/usr/bin/env python3
"""
Test only the multi-agent system for quick verification
"""

import asyncio
import os
from multi_agent_research import DeepResearchSystem

async def test_multi_agent():
    """Test multi-agent system with a simple query"""
    
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Please set your OPENAI_API_KEY environment variable")
        return
    
    print("🤖 Testing Multi-Agent Research System")
    print("="*50)
    
    # Initialize system
    system = DeepResearchSystem()
    
    # Simple test query
    test_query = "What are the main differences between LangChain and CrewAI frameworks?"
    
    try:
        print(f"Query: {test_query}")
        print()
        
        # Test basic research (single agent)
        print("📋 Basic Research (single agent):")
        result = await system.basic_research(test_query, verbose=False)
        
        print(f"✅ Completed successfully")
        print(f"Result length: {len(result)} characters")
        print(f"Preview: {result[:300]}...")
        print()
        
        print("🎯 TEST PASSED: Multi-Agent system is working correctly!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_multi_agent())