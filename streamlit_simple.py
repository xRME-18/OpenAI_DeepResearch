#!/usr/bin/env python3
"""
Simplified Streamlit App for OpenAI Deep Research API
Focuses on Deep Research API only for better Streamlit Cloud compatibility
"""

import streamlit as st
import asyncio
import time
from typing import Optional
import os
import json

# Try importing OpenAI
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    st.error("OpenAI package not available!")

# Page configuration
st.set_page_config(
    page_title="OpenAI Deep Research",
    page_icon="🔬",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .research-result {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def perform_deep_research(query: str, api_key: str, model: str = "o3-deep-research-2025-06-26") -> dict:
    """Perform research using OpenAI Deep Research API directly"""
    
    client = OpenAI(api_key=api_key, timeout=600.0)
    
    # Prepare the input messages
    input_messages = [
        {
            "role": "developer", 
            "content": [{"type": "input_text", "text": "You are a professional researcher. Provide comprehensive, well-cited research with specific sources."}]
        },
        {
            "role": "user", 
            "content": [{"type": "input_text", "text": query}]
        }
    ]
    
    # Configure tools
    tools = [
        {"type": "web_search_preview"},
        {"type": "code_interpreter", "container": {"type": "auto", "file_ids": []}}
    ]
    
    try:
        # Try with reasoning first
        response = client.responses.create(
            model=model,
            input=input_messages,
            tools=tools,
            reasoning={"summary": "auto"}
        )
    except Exception as e:
        if "must be verified" in str(e):
            # Fallback without reasoning
            response = client.responses.create(
                model=model,
                input=input_messages,
                tools=tools
            )
        else:
            raise e
    
    # Extract result
    final_output = response.output[-1].content[0].text
    
    # Extract citations
    citations = []
    if hasattr(response.output[-1].content[0], 'annotations'):
        for annotation in response.output[-1].content[0].annotations:
            if hasattr(annotation, 'start_index'):
                citations.append({
                    'title': getattr(annotation, 'title', 'Unknown Title'),
                    'url': getattr(annotation, 'url', ''),
                    'start_index': annotation.start_index,
                    'end_index': annotation.end_index,
                    'excerpt': final_output[annotation.start_index:annotation.end_index] if annotation.start_index < len(final_output) else ""
                })
    
    return {
        'text': final_output,
        'citations': citations,
        'model': model
    }

def main():
    # Header
    st.markdown('<div class="main-header">🔬 OpenAI Deep Research</div>', unsafe_allow_html=True)
    st.markdown("**Professional research using OpenAI's Deep Research API**")
    
    if not OPENAI_AVAILABLE:
        st.error("OpenAI package is required but not available.")
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key
        api_key = st.text_input(
            "OpenAI API Key:", 
            type="password",
            help="Your OpenAI API key (starts with 'sk-')"
        )
        
        if api_key and not api_key.startswith('sk-'):
            st.warning("⚠️ OpenAI API keys typically start with 'sk-'")
        
        # Model selection
        model = st.selectbox(
            "Deep Research Model:",
            [
                "o3-deep-research-2025-06-26",
                "o4-mini-deep-research-2025-06-26"
            ],
            help="Choose the research model"
        )
        
        st.info("💡 **About Deep Research API:**\n\n"
                "• Comprehensive research reports\n"
                "• Professional citations\n"
                "• Web search integration\n"
                "• Takes 2-5 minutes for quality results")
    
    # Main interface
    if not api_key:
        st.warning("👈 Please enter your OpenAI API key to get started.")
        
        # Examples
        st.subheader("🔍 Example Research Questions")
        examples = [
            "What are the latest developments in LLM orchestration frameworks?",
            "Current trends in AI agent deployment strategies",
            "Best practices for prompt engineering with GPT-4",
            "Comparison of vector databases for RAG applications",
            "How to implement error handling in OpenAI API calls?"
        ]
        
        for example in examples:
            st.write(f"• {example}")
        
        return
    
    # Research interface
    st.subheader("💬 Research Query")
    
    query = st.text_area(
        "What would you like to research?",
        placeholder="Enter your research question...\n\nExample: 'What are the top 3 open-source LLM frameworks for production deployment in 2025?'",
        height=120
    )
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        research_button = st.button(
            "🔬 Start Deep Research", 
            type="primary",
            disabled=not query.strip(),
            use_container_width=True
        )
    
    with col2:
        if st.button("🧹 Clear", use_container_width=True):
            st.rerun()
    
    # Perform research
    if research_button and query.strip():
        # Progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            status_text.text("🔬 Initializing Deep Research API...")
            progress_bar.progress(20)
            
            status_text.text("🔍 Performing comprehensive research...")
            progress_bar.progress(50)
            
            # Start timing
            start_time = time.time()
            
            # Perform research
            result = perform_deep_research(query.strip(), api_key, model)
            
            # Calculate timing
            duration = time.time() - start_time
            
            progress_bar.progress(100)
            status_text.text("✅ Research completed!")
            
            # Display results
            st.markdown('<div class="research-result">', unsafe_allow_html=True)
            
            # Header
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.markdown("### 📄 Research Results")
            with col2:
                st.metric("Duration", f"{duration:.1f}s")
            with col3:
                st.metric("Citations", len(result['citations']))
            
            # Main content
            st.markdown(result['text'])
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Citations
            if result['citations']:
                st.markdown("### 📚 Citations")
                for i, citation in enumerate(result['citations'], 1):
                    with st.expander(f"Citation {i}: {citation['title'][:60]}..."):
                        st.markdown(f"**[{citation['title']}]({citation['url']})**")
                        if citation['excerpt']:
                            st.markdown(f"*Excerpt: {citation['excerpt']}*")
            
            # Download
            download_content = f"""# Deep Research Results

**Query:** {query}
**Model:** {result['model']}
**Duration:** {duration:.1f}s
**Citations:** {len(result['citations'])}

## Results:
{result['text']}

## Citations:
"""
            for i, citation in enumerate(result['citations'], 1):
                download_content += f"\n{i}. [{citation['title']}]({citation['url']})\n   {citation['excerpt']}\n"
            
            st.download_button(
                "⬇️ Download Results",
                download_content,
                file_name=f"deep_research_{int(time.time())}.md",
                mime="text/markdown"
            )
            
        except Exception as e:
            progress_bar.progress(0)
            status_text.empty()
            st.error(f"❌ Research failed: {str(e)}")
            
            with st.expander("🔧 Troubleshooting"):
                st.markdown("""
                **Common issues:**
                - **Invalid API key**: Check your OpenAI API key
                - **Timeout**: Deep Research can take 2-5 minutes
                - **Organization limits**: Some features need verified accounts
                - **Rate limits**: Check your OpenAI usage limits
                """)

if __name__ == "__main__":
    main()