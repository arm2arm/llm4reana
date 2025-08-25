import streamlit as st
from crewai import Agent, Task, Crew, LLM

# Set up the Streamlit interface
st.title("HPC Blog Writing Assistant with CrewAI and Reana")
st.write("Create compelling blog posts on HPC topics using AI agents")

# Initialize session state for conversation history
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

# Initialize session state for content
if 'blog_content' not in st.session_state:
    st.session_state.blog_content = None

# Sidebar for conversation history
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

# Topic input
topic = st.text_input("Enter HPC topic for blog post", st.session_state.get('current_topic', "Optimizing GPU Performance in HPC Clusters"))

# Initialize Ollama LLM with specific model as requested
llm = LLM(model="ollama/qwen3-coder:latest", base_url="http://141.33.165.45:11434")

# Create agents for HPC content
research_agent = Agent(
    role='HPC Research Specialist',
    goal='Research and gather relevant information about HPC topics, best practices, and current trends',
    backstory='You are an expert in High Performance Computing with deep knowledge of HPC architectures, optimization techniques, and industry trends.',
    verbose=True,
    allow_delegation=False,
    llm=llm
)

writer_agent = Agent(
    role='Technical Blog Writer',
    goal='Write engaging, well-structured blog posts on HPC topics with proper technical accuracy',
    backstory='You are a skilled technical writer who can translate complex HPC concepts into accessible and engaging blog content.',
    verbose=True,
    allow_delegation=False,
    llm=llm
)

editor_agent = Agent(
    role='HPC Content Editor',
    goal='Review, edit, and optimize blog content for clarity, technical accuracy, and publishing readiness',
    backstory='You are an experienced editor with expertise in technical writing and HPC content optimization.',
    verbose=True,
    allow_delegation=False,
    llm=llm
)

# Create tasks for HPC blog post generation
research_task = Task(
    description=f"Research the topic '{topic}' for an HPC blog post. Gather current best practices, technical details, and relevant trends in the field.",
    expected_output="A comprehensive research report with key findings, technical details, and relevant references for the HPC topic",
    agent=research_agent
)

writing_task = Task(
    description=f"Write a well-structured blog post on '{topic}' based on the research findings. Include an introduction, main content sections, and practical tips.",
    expected_output="A complete, engaging blog post in markdown format with proper structure and technical accuracy",
    agent=writer_agent
)

editing_task = Task(
    description="Review and edit the blog post for clarity, grammar, technical accuracy, and publishing readiness. Ensure the content flows well and is suitable for publication.",
    expected_output="A polished, ready-to-publish blog post in markdown format with improved structure and clarity",
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
    with st.spinner("Generating HPC blog post..."):
        result = hpc_crew.kickoff()
        
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
    "Optimizing GPU Performance in HPC Clusters",
    "Best Practices for HPC Network Topology Design",
    "Memory Management Techniques in Large-Scale HPC Systems",
    "Parallel Programming Patterns for Modern HPC Architectures",
    "Energy Efficiency Strategies in HPC Data Centers"
]

for example_topic in example_topics:
    if st.button(f"Use Example: {example_topic}"):
        topic = example_topic
        # Update the session state to reflect the new topic
        st.session_state.current_topic = topic

st.write("---")
st.write("Note: This application requires Ollama to be running locally with the 'qwen3-coder:latest' model installed.")
