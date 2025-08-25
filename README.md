# Agentic experiements
We did 2 experiments using only 1 local hosted LLM: cline+`qwen3-coder:latest` + **context7** :
- Generate an CrewAI HPC blog writer Agent+ streamlit UI
- create a reana workflow+dockerfile for running the workflow from dockercontainer, ensure reana-client works on Win11 via docker

# HPC Blog Writing Assistant with CrewAI and Reana

This application provides an HPC blog writing assistant that uses CrewAI agents to research, write, and edit technical blog posts on High Performance Computing topics.

## Features

- Research, write, and edit technical blog posts on HPC topics
- Automated content generation by specialized AI agents:
  - HPC Research Specialist
  - Technical Blog Writer
  - HPC Content Editor
- Integration with Ollama for LLM capabilities
- Workflow management with Reana

## Requirements

- Python 3.8+
- Ollama installed locally with the `qwen3-coder:latest` model
- Streamlit
- CrewAI
- Pandas, Matplotlib, Seaborn, NumPy

## Installation

1. Install Ollama from [https://ollama.com/download](https://ollama.com/download)
2. Pull the Llama3 model:
   ```
   ollama pull llama3
   ```
3. Install Python dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Run the Streamlit application:
   ```
   streamlit run main.py
   ```
2. Enter an HPC topic for blog post generation
3. Click "Generate Blog Post" to create your content

## Data Format

This application does not require CSV files as it generates blog content based on topic inputs rather than analyzing datasets.

## Agents

1. **HPC Research Specialist**: Researches and gathers relevant information about HPC topics, best practices, and current trends
2. **Technical Blog Writer**: Writes engaging, well-structured blog posts on HPC topics with proper technical accuracy
3. **HPC Content Editor**: Reviews, edits, and optimizes blog content for clarity, technical accuracy, and publishing readiness

## Note

This application requires Ollama to be running locally with the 'qwen3-coder:latest' model installed.

## Docker Usage

To build and run this application using Docker:

### For Streamlit Blogger App:
1. Build the Docker image:
   ```
   docker build -t hpc-blog-assistant -f Dockerfile .
   ```

2. Run the Docker container:
   ```
   docker run -p 8501:8501 hpc-blog-assistant
   ```

3. Access the application at http://localhost:8501

### For REANA Client:
1. Remove existing Docker image (if it exists):
   ```
   docker rmi -f reana-client || true
   ```
2. Build the Docker image from scratch:
   ```
   docker build -t reana-client -f reana.Dockerfile .
   ```

3. Run the Docker container in interactive mode with environment variables and local folder mount bind:

   ```
   # Set REANA environment variables
   export REANA_SERVER_URL=https://reana-p4n.aip.de
   export REANA_ACCESS_TOKEN=k_gsk84cufvma58pv832nw
   
   # For Linux/macOS:
   docker run -it --env REANA_SERVER_URL=https://reana-p4n.aip.de --env REANA_ACCESS_TOKEN=k_gsk84cufvma58pv832nw --volume /local/path/to/data:/data reana-client
   ```
   # For Windows (binding
   D:\Users\arm2arm\Projects\LLM\agents\local-agent):

```
docker run -it --env REANA_SERVER_URL=https://reana-p4n.aip.de --env REANA_ACCESS_TOKEN=k_gsk84cufvma58pv832nw --volume D:/Users/arm2arm/Projects/LLM/agents/local-agent:/data reana-client
   ```

## Running REANA Client with reana.yaml

To run the REANA client with the provided `reana.yaml` configuration:

1. First, ensure you have built the REANA client Docker image:
   ```
   docker build -t reana-client -f reana.Dockerfile .
   ```

2. Set the required environment variables:
   ```
   export REANA_SERVER_URL=https://reana-p4n.aip.de
   export REANA_ACCESS_TOKEN=k_gsk84cufvma58pv832nw
   ```

3. Run the workflow using the reana.yaml configuration file:
   ```
   docker run -it --env REANA_SERVER_URL=https://reana-p4n.aip.de --env REANA_ACCESS_TOKEN=k_gsk84cufvma58pv832nw --volume $(pwd):/data reana-client
   ```

   For Windows users:
   ```
   docker run -it --env REANA_SERVER_URL=https://reana-p4n.aip.de --env REANA_ACCESS_TOKEN=k_gsk84cufvma58pv832nw --volume D:/Users/arm2arm/Projects/LLM/agents/local-agent:/data reana-client
   ```

   Then inside the container, execute:
   ```
   reana-client run -f reana.yaml
   ```

   This approach will:
   - Use the current directory as the working directory for the workflow
   - Execute the workflow defined in `reana.yaml`
   - Mount the local directory to `/data` inside the container
   - Submit the workflow to the REANA server

   Alternative single-command approach (if the above doesn't work):
   ```
   docker run -it --env REANA_SERVER_URL=https://reana-p4n.aip.de --env REANA_ACCESS_TOKEN=xxxx --volume $(pwd):/data reana-client reana-client run -f reana.yaml
   ```
