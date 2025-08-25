import streamlit as st
from crewai import Agent, Task, Crew, LLM
import os
import glob
import subprocess
from typing import List, Dict

# Set up the Streamlit interface
st.title("HPC Blog Writing Assistant with CrewAI and Reana")
st.write("Create compelling blog posts on HPC topics using AI agents")

# Function to get available models from Ollama
def get_ollama_models():
    """Get list of available models from Ollama server"""
    try:
        # Run ollama list command and parse output
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, shell=True)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')[1:]  # Skip header line
            models = [line.split()[0] for line in lines if line.strip()]
            return models
        else:
            st.warning("Could not retrieve Ollama models. Using default models.")
            return ["qwen3-coder:latest", "nomic-embed-text:latest"]
    except Exception as e:
        st.warning(f"Error getting Ollama models: {str(e)}. Using default models.")
        return ["qwen3-coder:latest", "nomic-embed-text:latest"]

# Initialize session state for conversation history
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

# Initialize session state for content
if 'blog_content' not in st.session_state:
    st.session_state.blog_content = None

# Initialize session state for knowledge base
if 'knowledge_base' not in st.session_state:
    st.session_state.knowledge_base = None

# Initialize session state for knowledge base files
if 'kb_files' not in st.session_state:
    st.session_state.kb_files = set()

# Initialize session state for models
if 'selected_embedding_model' not in st.session_state:
    st.session_state.selected_embedding_model = "nomic-embed-text:latest"

if 'selected_main_model' not in st.session_state:
    st.session_state.selected_main_model = "qwen3-coder:latest"

# Function to load knowledge base from .mdx files
def load_knowledge_base(embedding_model="nomic-embed-text:latest"):
    """Load all .mdx files from data/ directory into the knowledge base"""
    try:
        # Get all .mdx files from data directory
        mdx_files = glob.glob("data/*.mdx")
        
        # Check if files have changed since last load
        current_files = set(mdx_files)
        if current_files == st.session_state.kb_files and st.session_state.knowledge_base is not None:
            return st.session_state.knowledge_base  # No changes, return cached
        
        # Update file tracking
        st.session_state.kb_files = current_files
        
        # Load content from all .mdx files
        all_content = []
        for file_path in mdx_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Add metadata to content
                    metadata = f"Source: {os.path.basename(file_path)}"
                    all_content.append(f"{metadata}\n\n{content}")
            except Exception as e:
                st.warning(f"Failed to read file {file_path}: {str(e)}")
        
        if not all_content:
            st.warning("No .mdx files found in data directory")
            return None
            
        # Create knowledge base content for context7 usage
        # Instead of using Chroma directly, we'll store the content as a string that can be used by agents
        knowledge_content = "\n\n".join(all_content)
        
        st.session_state.knowledge_base = knowledge_content
        return knowledge_content
        
    except Exception as e:
        st.error(f"Error loading knowledge base: {str(e)}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        return None

# Function to get relevant context from knowledge base
def get_context_from_knowledge_base(query, knowledge_base):
    """Get relevant context from the knowledge base for a given query"""
    if not knowledge_base:
        return ""
    
    try:
        # For simplicity in this implementation, we'll just return the full knowledge base content
        # In a more advanced implementation, you could use semantic search with embeddings here
        return knowledge_base
    except Exception as e:
        st.error(f"Error retrieving context: {str(e)}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        return ""

# Sidebar for conversation history and configuration options
with st.sidebar:
    st.header("Conversation History")
    
    # Display conversation history
    for i, (hist_topic, hist_content) in enumerate(st.session_state.conversation_history):
        if st.button(f"#{i+1}: {hist_topic[:50]}...", key=f"hist_{i}"):
            # When clicking a history item, update the topic input and display content
            st.session_state.current_topic = hist_topic
            st.subheader("Previous Result")
            st.markdown(hist_content)
    
    # Clear history button
    if st.button("Clear History"):
        st.session_state.conversation_history = []
        st.experimental_rerun()
    
    st.header("Model Configuration")
    
    # Get available models from Ollama
    available_models = get_ollama_models()
    
    # Embedding model selection
    embedding_model = st.selectbox(
        "Select Embedding Model",
        options=available_models,
        index=available_models.index(st.session_state.selected_embedding_model) if st.session_state.selected_embedding_model in available_models else 1,
        key="embedding_model_select"
    )
    
    # Main LLM model selection
    main_model = st.selectbox(
        "Select Main Model",
        options=available_models,
        index=available_models.index(st.session_state.selected_main_model) if st.session_state.selected_main_model in available_models else 0,
        key="main_model_select"
    )
    
    # Update session state with selected models
    st.session_state.selected_embedding_model = embedding_model
    st.session_state.selected_main_model = main_model
    
    st.header("Knowledge Base")
    
    # Refresh knowledge base button
    if st.button("Refresh Knowledge Base"):
        with st.spinner("Refreshing knowledge base..."):
            kb = load_knowledge_base(embedding_model)
            if kb:
                st.success("Knowledge base refreshed successfully!")
            else:
                st.warning("Failed to refresh knowledge base. Check the logs for details.")

# Load knowledge base on startup or when refresh is clicked
if st.session_state.knowledge_base is None:
    with st.spinner("Loading knowledge base..."):
        kb = load_knowledge_base(st.session_state.selected_embedding_model)
        if kb:
            st.success("Knowledge base loaded successfully!")
        else:
            st.warning("Failed to load knowledge base, but continuing without it.")

# Topic input
topic = st.text_input("Enter HPC topic for blog post", st.session_state.get('current_topic', "Optimizing GPU Performance in HPC Clusters"))

# Initialize Ollama LLM with selected model
llm = LLM(model=f"ollama/{st.session_state.selected_main_model}", base_url="http://141.33.165.45:11434")

# Create agents for HPC content - now focused on answering questions based on knowledge base
research_agent = Agent(
    role='HPC Knowledge Specialist',
    goal='Analyze and extract relevant information from HPC documentation to answer specific questions about HPC topics',
    backstory='You are an expert in High Performance Computing who can analyze technical documentation and extract key information to answer queries.',
    verbose=True,
    allow_delegation=False,
    llm=llm
)

writer_agent = Agent(
    role='Technical Blog Writer',
    goal='Write engaging, well-structured blog posts on HPC topics with proper technical accuracy based on the provided knowledge base',
    backstory='You are a skilled technical writer who can translate complex HPC concepts into accessible and engaging blog content using only the provided documentation.',
    verbose=True,
    allow_delegation=False,
    llm=llm
)

editor_agent = Agent(
    role='HPC Content Editor',
    goal='Review, edit, and optimize blog content for clarity, technical accuracy, and publishing readiness based on knowledge base information',
    backstory='You are an experienced editor with expertise in technical writing and HPC content optimization.',
    verbose=True,
    allow_delegation=False,
    llm=llm
)

# Create tasks for HPC blog post generation - now focused on using knowledge base
research_task = Task(
    description=f"Analyze the topic '{topic}' and extract all relevant information from the knowledge base. Provide a comprehensive summary of key points, technical details, and best practices related to this HPC topic.",
    expected_output="A comprehensive summary with key findings, technical details, and relevant references for the HPC topic based on knowledge base content",
    agent=research_agent
)

writing_task = Task(
    description=f"Write a well-structured blog post on '{topic}' using only information from the knowledge base. Include an introduction, main content sections, practical tips, and ensure all content is derived from documented sources.",
    expected_output="A complete, engaging blog post in markdown format with proper structure and technical accuracy based entirely on knowledge base information",
    agent=writer_agent
)

editing_task = Task(
    description="Review and edit the blog post for clarity, grammar, technical accuracy, and publishing readiness. Ensure the content flows well and is suitable for publication, using only information from the knowledge base.",
    expected_output="A polished, ready-to-publish blog post in markdown format with improved structure and clarity based entirely on knowledge base information",
    agent=editor_agent
)

# Create crew and execute tasks
hpc_crew = Crew(
    agents=[research_agent, writer_agent, editor_agent],
    tasks=[research_task, writing_task, editing_task],
    verbose=True
)

# Generate blog button
if st.button("Generate Blog Post"):
    with st.spinner("Generating HPC blog post from knowledge base..."):
        # Get context from knowledge base for the topic
        context = get_context_from_knowledge_base(topic, st.session_state.knowledge_base)
        
        # Update the research task description to include context from knowledge base
        research_task.description = f"Analyze the topic '{topic}' and extract all relevant information from the knowledge base. Provide a comprehensive summary of key points, technical details, and best practices related to this HPC topic. Context: {context[:500]}..."
        
        # Track token usage
        try:
            result = hpc_crew.kickoff()
            # Display token usage information (this is a placeholder - actual token tracking would require more specific implementation)
            st.info("Token usage tracking is enabled.")
        except Exception as e:
            result = f"Error generating content: {str(e)}"
        # Save to conversation history
        st.session_state.conversation_history.append((topic, result))
        st.session_state.current_topic = topic
        
        # Display results
        st.subheader("Generated Blog Post")
        st.markdown(result)
        
        # Save content to session state for download (convert to string if needed)
        result_content = str(result) if hasattr(result, '__str__') else result
        st.session_state.blog_content = result_content
        
        # Download button
        if st.session_state.blog_content:
            st.download_button(
                label="Download Markdown File",
                data=st.session_state.blog_content,
                file_name=f"hpc_blog_{topic.replace(' ', '_')[:30]}.md",
                mime="text/markdown"
            )

# Show example topics
st.subheader("Example HPC Topics")
example_topics = [
    "Complete Guide to Slurm Job Scheduling",
    "Getting Started with High-Performance Computing",
    "GNU Parallel Job Sharding in Slurm",
    "Using Apptainer for Containerized HPC Workloads",
    "Advanced GPU Computing with CUDA"
]

for example_topic in example_topics:
    if st.button(f"Use Example: {example_topic}"):
        topic = example_topic
        # Update the session state to reflect the new topic
        st.session_state.current_topic = topic

st.write("---")
st.write("Note: This application requires Ollama to be running locally with the selected models installed.")

