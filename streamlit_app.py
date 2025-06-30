#!/usr/bin/env python3
"""
Streamlit Web App for OpenAI Deep Research
Provides a user-friendly interface for the OpenAI research system
"""

import streamlit as st
import asyncio
import os
import time
from typing import Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import your research modules with fallback handling
MODULES_AVAILABLE = {
    'interface': False,
    'deep_api': False,
    'agents': False
}

try:
    from openai_research_interface import ResearchInterface, ResearchMethod
    MODULES_AVAILABLE['interface'] = True
except ImportError as e:
    st.warning(f"Research interface not available: {e}")

try:
    from openai_deep_research_api import OpenAIDeepResearchAPI
    MODULES_AVAILABLE['deep_api'] = True
except ImportError as e:
    st.warning(f"Deep Research API not available: {e}")

try:
    from openai_agents_research import DeepResearchSystem
    MODULES_AVAILABLE['agents'] = True
except ImportError as e:
    st.warning(f"OpenAI Agents not available: {e}")

# If no modules are available, show error and basic functionality
if not any(MODULES_AVAILABLE.values()):
    st.error("❌ No research modules available. Please check the deployment.")
    st.info("This may be due to missing dependencies. The app requires openai-agents package which may not be available on Streamlit Cloud.")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="OpenAI Deep Research",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .research-result {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .citation-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 5px;
        border-left: 3px solid #ffc107;
        margin: 0.5rem 0;
    }
    .method-badge {
        background-color: #28a745;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.9rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<div class="main-header">🚀 OpenAI Deep Research</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Professional research using OpenAI\'s Agents SDK and Deep Research API</div>', unsafe_allow_html=True)
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key input
        api_key = st.text_input(
            "OpenAI API Key:",
            type="password",
            help="Your OpenAI API key (starts with 'sk-'). This is not stored and only used for your research session."
        )
        
        if api_key and not api_key.startswith('sk-'):
            st.warning("⚠️ OpenAI API keys typically start with 'sk-'")
        
        st.divider()
        
        # Method selection based on available modules
        st.subheader("🤖 Research Method")
        
        available_methods = []
        method_map = {}
        
        if MODULES_AVAILABLE['interface']:
            available_methods.append("Auto-select (Recommended)")
            method_map["Auto-select (Recommended)"] = ResearchMethod.AUTO
            
        if MODULES_AVAILABLE['agents']:
            available_methods.append("OpenAI Agents (Fast, 30-60s)")
            method_map["OpenAI Agents (Fast, 30-60s)"] = ResearchMethod.OPENAI_AGENTS
            
        if MODULES_AVAILABLE['deep_api']:
            available_methods.append("Deep Research API (Comprehensive, 2-5min)")
            method_map["Deep Research API (Comprehensive, 2-5min)"] = ResearchMethod.DEEP_RESEARCH_API
        
        if not available_methods:
            st.error("No research methods available!")
            st.stop()
            
        method_choice = st.selectbox(
            "Choose research approach:",
            available_methods,
            help="Available methods based on successfully loaded modules."
        )
        
        selected_method = method_map[method_choice]
        
        st.divider()
        
        # Advanced options
        with st.expander("🔧 Advanced Options"):
            if selected_method in [ResearchMethod.DEEP_RESEARCH_API, ResearchMethod.AUTO]:
                deep_model = st.selectbox(
                    "Deep Research Model:",
                    [
                        "o3-deep-research-2025-06-26 (Most capable)",
                        "o4-mini-deep-research-2025-06-26 (Faster)"
                    ]
                )
                model_name = deep_model.split(" ")[0]
            else:
                model_name = "o3-deep-research-2025-06-26"
            
            custom_system = st.text_area(
                "Custom System Message:",
                placeholder="Optional: Custom instructions for the research (e.g., 'Focus on technical implementation details')",
                height=100
            )
            
            verbose_mode = st.checkbox("Verbose Mode", value=True, help="Show detailed progress information")
        
        # Information
        st.divider()
        st.info("💡 **Tips:**\n\n"
                "• Use **Auto-select** for best results\n\n"
                "• **OpenAI Agents** for quick technical questions\n\n"
                "• **Deep Research** for comprehensive analysis\n\n"
                "• Try specific queries for better results")
    
    # Main content area
    if not api_key:
        st.warning("👈 Please enter your OpenAI API key in the sidebar to get started.")
        
        # Show example and info
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔍 What can you research?")
            st.markdown("""
            **Technical Questions:**
            - "How to implement error handling in LangChain?"
            - "Best practices for LLM prompt engineering"
            - "Python async/await performance optimization"
            
            **Market Analysis:**
            - "Current landscape of AI agent frameworks"
            - "Trends in open-source LLM tools 2025"
            - "Competitive analysis of vector databases"
            
            **Academic Research:**
            - "Recent advances in transformer architectures"
            - "Ethics considerations in AI deployment"
            - "Sustainability of large language models"
            """)
        
        with col2:
            st.subheader("🚀 Features")
            st.markdown("""
            **Dual OpenAI Approach:**
            - Custom OpenAI Agents orchestration
            - Native Deep Research API integration
            - Intelligent method selection
            
            **Professional Output:**
            - Rich citations with source links
            - Structured research findings
            - Real-time progress tracking
            
            **Production Ready:**
            - Error handling and fallbacks
            - Secure API key handling
            - Scalable architecture
            """)
        
        return
    
    # Research interface
    st.subheader("💬 Research Query")
    
    # Query input
    query = st.text_area(
        "What would you like to research?",
        placeholder="Enter your research question here...\n\nExample: 'What are the latest developments in LLM orchestration frameworks for production deployment?'",
        height=120,
        help="Be specific for better results. The system works best with focused, well-defined questions."
    )
    
    # Research button and progress
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        research_button = st.button(
            "🔍 Start Research", 
            type="primary", 
            disabled=not query.strip(),
            use_container_width=True
        )
    
    with col2:
        if st.button("📝 Example Queries", use_container_width=True):
            examples = [
                "What are the top 3 open-source LLM orchestration frameworks in 2025?",
                "How do I implement error handling in OpenAI API calls?",
                "Current trends in AI agent deployment strategies",
                "Best practices for prompt engineering with GPT-4",
                "Comparison of vector databases for RAG applications"
            ]
            st.info("**Example queries:**\n\n" + "\n\n".join([f"• {ex}" for ex in examples]))
    
    with col3:
        if st.button("🧹 Clear", use_container_width=True):
            st.rerun()
    
    # Perform research
    if research_button and query.strip():
        perform_research(query.strip(), api_key, selected_method, model_name, custom_system, verbose_mode)

def perform_research(query: str, api_key: str, method: ResearchMethod, model_name: str, custom_system: str, verbose: bool):
    """Perform the research and display results"""
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.text("🔧 Initializing research system...")
        progress_bar.progress(10)
        
        # Initialize research interface
        interface_kwargs = {"deep_research_model": model_name} if model_name else {}
        interface = ResearchInterface(
            method=method,
            api_key=api_key,
            **interface_kwargs
        )
        
        status_text.text("🚀 Starting research...")
        progress_bar.progress(25)
        
        # Prepare research parameters
        research_kwargs = {}
        if custom_system.strip():
            research_kwargs["system_message"] = custom_system.strip()
        
        # Start timing
        start_time = time.time()
        
        # Show method being used
        if method == ResearchMethod.AUTO:
            predicted_method = interface._auto_select_method(query)
            st.info(f"🤖 Auto-selected method: **{predicted_method.value}**")
        
        status_text.text(f"🔍 Researching using {method.value}...")
        progress_bar.progress(50)
        
        # Perform research (run async function)
        result = asyncio.run(interface.research(
            query=query,
            verbose=verbose,
            **research_kwargs
        ))
        
        # Calculate timing
        end_time = time.time()
        duration = end_time - start_time
        
        progress_bar.progress(100)
        status_text.text("✅ Research completed!")
        
        # Display results
        display_results(result, duration)
        
    except Exception as e:
        progress_bar.progress(0)
        status_text.empty()
        st.error(f"❌ Research failed: {str(e)}")
        
        # Show troubleshooting tips
        with st.expander("🔧 Troubleshooting"):
            st.markdown("""
            **Common issues:**
            - **Invalid API key**: Make sure your OpenAI API key is correct and active
            - **Timeout**: Deep Research API can take 2-5 minutes, please wait
            - **Organization not verified**: Some features require verified OpenAI accounts
            - **Rate limits**: You may have hit OpenAI's usage limits
            
            **Solutions:**
            - Check your API key format (should start with 'sk-')
            - Try the OpenAI Agents method for faster results
            - Wait a few minutes and try again
            - Check your OpenAI account billing and usage
            """)

def display_results(result, duration: float):
    """Display research results in a formatted way"""
    
    # Result header
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown(f'<span class="method-badge">{result.method_used.replace("_", " ").title()}</span>', 
                   unsafe_allow_html=True)
    
    with col2:
        st.metric("Duration", f"{duration:.1f}s")
    
    with col3:
        citations_count = result.metadata.get('citations_count', 0)
        st.metric("Citations", citations_count)
    
    # Main research result
    st.markdown('<div class="research-result">', unsafe_allow_html=True)
    st.markdown("### 📄 Research Results")
    st.markdown(result.result)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Metadata and citations
    if result.metadata:
        with st.expander("📊 Research Metadata", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.json({k: v for k, v in result.metadata.items() if k != 'citations'})
            
            with col2:
                if result.method_used == "openai_agents":
                    agents_used = result.metadata.get('agents_used', [])
                    st.write("**Agents Used:**")
                    for agent in agents_used:
                        st.write(f"• {agent.title()}")
        
        # Citations section
        if result.metadata.get('citations'):
            st.markdown("### 📚 Citations")
            
            citations = result.metadata['citations']
            for i, citation in enumerate(citations, 1):
                with st.container():
                    st.markdown(f'<div class="citation-box">', unsafe_allow_html=True)
                    st.markdown(f"**{i}. [{citation['title']}]({citation['url']})**")
                    if citation.get('excerpt'):
                        st.markdown(f"*{citation['excerpt']}*")
                    st.markdown('</div>', unsafe_allow_html=True)
    
    # Action buttons
    st.divider()
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 New Research", type="secondary", use_container_width=True):
            st.rerun()
    
    with col2:
        # Download results
        download_content = f"""# Research Results

**Query:** {result.query}
**Method:** {result.method_used}
**Duration:** {duration:.1f}s

## Results:
{result.result}

## Metadata:
{result.metadata}
"""
        st.download_button(
            "⬇️ Download Results",
            download_content,
            file_name=f"research_results_{int(time.time())}.md",
            mime="text/markdown",
            use_container_width=True
        )
    
    with col3:
        # Share functionality
        if st.button("📤 Share Query", use_container_width=True):
            share_url = f"https://your-streamlit-app.com/?query={result.query}"
            st.code(share_url)
            st.success("Share this URL to let others research the same query!")

if __name__ == "__main__":
    main()