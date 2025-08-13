# Vote Guard

Vote Guard is a sophisticated multi-agent system designed to analyze social media posts for potential misinformation, bias, and sentiment. Built using LangGraph, this application coordinates multiple specialized AI agents to provide comprehensive content analysis, helping users critically evaluate the information they encounter online, particularly in political and social discourse.

## Features

- **Misinformation Detection**: Identifies potentially false or misleading claims in social media posts
- **Bias Analysis**: Detects political or ideological bias in the content
- **Sentiment Analysis**: Evaluates the emotional tone of the post
- **User-friendly Interface**: Simple web interface built with Streamlit
- **Multi-Agent Architecture**: Employs specialized AI agents working in coordination for comprehensive analysis
- **AI-Powered**: Utilizes state-of-the-art language models for accurate analysis
- **Modular Design**: Easy to extend with new analysis modules and agents

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Git (for cloning the repository)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Naveen-PJ/Vote-Guard.git
   cd Vote-Guard
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables (choose one method):

   **Option 1: Using .env file**
   Create a `.env` file in the project root and add your API keys:
   ```
   # Required API keys
   GROQ_API_KEY=your_groq_api_key_here
   HUGGINGFACE_TOKEN=your_huggingface_token_here
   ```

   **Option 2: Using Streamlit secrets**
   For deployment, create a `.streamlit/secrets.toml` file with:
   ```toml
   # Required API keys
   GROQ_API_KEY = "your_groq_api_key_here"
   HUGGINGFACE_TOKEN = "your_huggingface_token_here"
   ```
   
   > **Note**: The application will first check for `.env` file, and if not found, it will look for Streamlit secrets.

## Usage

1. Start the application:
   ```bash
   streamlit run app.py
   ```

2. Open your web browser and navigate to the URL shown in the terminal (usually http://localhost:8501)

3. Paste a social media post into the text area and click "Analyze Post"

4. View the analysis results, which will include:
   - Potential misinformation indicators
   - Bias detection
   - Sentiment analysis
   - Overall assessment

## Agent Architecture

Vote Guard implements a sophisticated agent-based workflow using LangGraph, where different components handle specific aspects of content analysis:

1. **Context Retriever**: Queries the ChromaDB vector database for similar past reports to provide context
2. **Analysis Engine**: A comprehensive agent that runs multiple ML models for:
   - Lie detection (misinformation analysis)
   - Sentiment analysis
   - Political bias detection
3. **Report Generator**: Synthesizes results from all analyses into a coherent report
4. **Memory Saver**: Stores the analysis results in the ChromaDB for future reference

The system uses a directed graph workflow (LangGraph) to manage the flow of data between these components, ensuring efficient and modular processing.

## Project Structure

```
Vote-Guard/
├── app.py                 # Main Streamlit application interface
├── agent.py               # Core agent definitions and LangGraph workflow
├── requirements.txt        # Python dependencies
├── memory/                # ChromaDB vector database files
├── models/                # Downloaded AI models (Hugging Face)
└── logs/                  # Application runtime logs
```

## Dependencies

- **Streamlit**: Web application framework for the user interface
- **LangGraph**: For building stateful, multi-actor applications with LLMs
- **Hugging Face Transformers**: For running pre-trained NLP models
- **ChromaDB**: Vector database for storing and retrieving analysis results
- **Sentence-Transformers**: For generating text embeddings
- **Groq**: High-performance LLM inference (via langchain-groq)
- **Python-dotenv**: For environment variable management

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

For support, please open an issue in the GitHub repository.