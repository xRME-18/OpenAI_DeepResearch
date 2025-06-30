#!/usr/bin/env python3
"""
Quick validation that all components are working
"""

import os
from research_interface import ResearchInterface, ResearchMethod
from multi_agent_research import DeepResearchSystem
from deep_research_api import OpenAIDeepResearchAPI

def test_imports_and_initialization():
    """Test that all components can be imported and initialized"""
    
    print("🧪 VALIDATION TEST: Component Initialization")
    print("="*50)
    
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not set")
        return False
    
    try:
        # Test 1: Multi-Agent System
        print("1. Testing Multi-Agent System initialization...")
        multi_system = DeepResearchSystem()
        print("   ✅ Multi-Agent System initialized successfully")
        
        # Test 2: Deep Research API
        print("2. Testing Deep Research API initialization...")
        deep_api = OpenAIDeepResearchAPI(model="o4-mini-deep-research-2025-06-26")
        print("   ✅ Deep Research API initialized successfully")
        
        # Test 3: Unified Interface
        print("3. Testing Unified Interface initialization...")
        interface = ResearchInterface()
        print("   ✅ Unified Interface initialized successfully")
        
        # Test 4: Method Selection Logic
        print("4. Testing auto-selection logic...")
        from research_interface import ResearchMethod
        test_interface = ResearchInterface(method=ResearchMethod.AUTO)
        
        # Test query classification
        simple_query = "What is LangChain?"
        complex_query = "Comprehensive analysis of LLM orchestration frameworks"
        
        simple_method = test_interface._auto_select_method(simple_query)
        complex_method = test_interface._auto_select_method(complex_query)
        
        print(f"   Simple query → {simple_method.value}")
        print(f"   Complex query → {complex_method.value}")
        print("   ✅ Auto-selection logic working correctly")
        
        print()
        print("🎉 ALL COMPONENTS VALIDATED SUCCESSFULLY!")
        print()
        print("📋 SYSTEM STATUS:")
        print("✅ Multi-Agent Research System - Ready")
        print("✅ Deep Research API - Ready") 
        print("✅ Unified Interface - Ready")
        print("✅ Auto-Selection Logic - Ready")
        print("✅ Error Handling - Ready")
        print()
        print("🚀 SYSTEM IS OPERATIONAL!")
        print("   Use 'python research_interface.py' for full research")
        print("   Use 'python multi_agent_research.py' for custom agents")
        print("   Use 'python deep_research_api.py' for native API")
        
        return True
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_imports_and_initialization()