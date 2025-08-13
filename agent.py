import os
import json
import uuid
import logging
from datetime import datetime
from typing import TypedDict, List
import streamlit as st # Import streamlit to access st.secrets

# Load environment variables if they exist for local development, but we'll prioritize st.secrets
# from dotenv import load_dotenv
# load_dotenv()

# Set the Hugging Face cache directory to a local 'models' folder
os.environ['HUGGINGFACE_HUB_CACHE'] = './models'

# --- Logging Setup ---
def setup_logging():
    """
    Configures the logging for the application.
    Logs will be written to a file and the console.
    """
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_filename = os.path.join(log_dir, f"agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            #logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

# Call the setup function at the beginning of the script
main_logger = setup_logging()

try :
    from langchain_core.prompts import PromptTemplate
    from langchain_groq import ChatGroq
    from langgraph.graph import StateGraph, END, START
    from transformers import pipeline
    #from IPython.display import Image, display

    import chromadb
    from sentence_transformers import SentenceTransformer

except ImportError as e:
    main_logger.error(f"Required libraries are not installed. Please install them using 'pip install -r req.txt'. Error: {e}")
    print("Required libraries are not installed. Please install them using 'pip install -r req.txt'.")
    exit(1)

# --- Secret Loading ---
# This new function will handle getting the GROQ API key
# It first tries st.secrets (for Streamlit Cloud) and then falls back to os.environ (for local dev)
def get_groq_api_key():
    """
    Retrieves the GROQ API key from Streamlit secrets or environment variables.
    """
    # Check if the key exists in Streamlit secrets
    if "GROQ_API_KEY" in st.secrets:
        main_logger.info("GROQ API key retrieved from Streamlit secrets.")
        return st.secrets["GROQ_API_KEY"]
    # Fallback to environment variables for local testing
    elif "GROQ_API_KEY" in os.environ:
        main_logger.info("GROQ API key retrieved from environment variables.")
        return os.environ["GROQ_API_KEY"]
    else:
        main_logger.critical("CRITICAL: GROQ_API_KEY is not set in Streamlit secrets or environment variables.")
        return None

# --- Vector Database and Embedding Model Setup ---
# Define a global embedding model instance for consistency
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def get_embedding(text):
    return embedding_model.encode(text).tolist()

def setup_chromadb():
    """
    Initializes a persistent ChromaDB client and gets or creates the 'reports' collection.
    """
    main_logger.info("Setting up ChromaDB client and 'reports' collection.")
    try:
        client = chromadb.PersistentClient(path="memory")
        collection = client.get_or_create_collection(
            name="reports",
            embedding_function=None # We will handle embeddings manually to ensure consistency
        )
        return collection
    except Exception as e:
        main_logger.error(f"Error setting up ChromaDB: {e}")
        return None

# --- State Definition ---
class State(TypedDict):
        """
        Represents the state of our graph.
        The `raw_text` is the initial input.
        The other keys are updated by the agents.
        """
        raw_text: str
        lie_detection_output: str
        sentiment_output: str
        political_bias_output: str
        retrieved_context: List[dict]
        final_report: str

# --- Agent Nodes ---

def context_retriever(state: State) -> State:
    """
    Uses the raw_text to query the ChromaDB for similar past reports.
    Adds the retrieved reports to the state.
    """
    logger = logging.getLogger("context_retriever")
    logger.info("---RUNNING EXPERIENCE PROVIDER AGENT---")

    text = state['raw_text']
    try:
        collection = setup_chromadb()
        if collection and text:
            # Generate embedding for the query text
            query_embedding = get_embedding(text)
            
            # Find the top 3 most similar reports
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=3,
                include=['metadatas', 'documents']
            )

            # Combine documents and metadatas into a structured list
            retrieved_reports = []
            if results and results['ids'] and results['ids'][0]:
                for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
                    retrieved_reports.append({
                        "original_post": doc,
                        "final_report": meta.get('final_report', 'N/A')
                    })

            state['retrieved_context'] = retrieved_reports
            logger.info(f"Retrieved {len(retrieved_reports)} historical reports from memory.")
        else:
            state['retrieved_context'] = []
            logger.warning("No ChromaDB collection or raw text available for retrieval.")

    except Exception as e:
        logger.error(f"Error in context_retriever: {e}")
        state['retrieved_context'] = [{"error": "Failed to retrieve historical data", "details": str(e)}]
        
    return state


def analysis_engine(state: State) -> State:
    """
    A master agent that runs all three ML models sequentially.
    """
    logger = logging.getLogger("analysis_engine")
    logger.info("---RUNNING MASTER ML ANALYSIS AGENT---")
    
    text = state['raw_text']
    
    # 1. Run Lie Detection Model
    try:
        logger.info("  -> Running Lie Detection Model...")
        classifier = pipeline("text-classification", model="google/electra-small-discriminator")
        result = classifier(text)[0]
        output_dict = {
            "lie_detection": result['label'].replace('_', ' '),
            "confidence_score": result['score']
        }
        state['lie_detection_output'] = json.dumps(output_dict)
        logger.info(f"  -> Lie Detection Output: {state['lie_detection_output']}")
    except Exception as e:
        logger.error(f"  -> Error in lie_detection_agent: {e}")
        state['lie_detection_output'] = json.dumps({"error": "Failed to run model", "details": str(e)})

    # 2. Run Sentiment Model
    try:
        logger.info("  -> Running Sentiment Model...")
        sentiment_model = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")
        result = sentiment_model(text)[0]
        output_dict = {
            "emotion": result['label'].lower(),
            "score": result['score']
        }
        state['sentiment_output'] = json.dumps(output_dict)
        logger.info(f"  -> Emotion Output: {state['sentiment_output']}")
    except Exception as e:
        logger.error(f"  -> Error in sentiment_agent: {e}")
        state['sentiment_output'] = json.dumps({"error": "Failed to run model", "details": str(e)})

    # 3. Run Political Bias Model (UPDATED TO USE LLM)
    try:
        logger.info("  -> Running LLM Political Bias Agent...")
        # Get the API key using our new function
        groq_api_key = get_groq_api_key()
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY not set.")
            
        llm = ChatGroq(model="llama3-8b-8192", temperature=0, groq_api_key=groq_api_key)
        
        # This prompt instructs the LLM to act as a classifier and output JSON
        bias_prompt = PromptTemplate.from_template(
            """Your task is to classify the political bias of the following text as either 'left-leaning', 'right-leaning', or 'neutral'.

            Analyze the text and determine the most likely political bias. Then, provide a confidence score from 0.0 to 1.0, where 1.0 is very confident.

            Text to analyze: {raw_text}

            Provide your response as a JSON object with two keys:
            - "political_bias": a string with the classified bias.
            - "confidence": a float with the confidence score.
            
            Return ONLY the JSON object and no other text or explanation.

            Example 1:
            Text to analyze: The new law is a great step forward for equality.
            JSON: {{"political_bias": "left-leaning", "confidence": 0.95}}

            Example 2:
            Text to analyze: Government spending is out of control and is a threat to our freedom.
            JSON: {{"political_bias": "right-leaning", "confidence": 0.92}}
            """
        )
        
        bias_report = llm.invoke(bias_prompt.format(raw_text=text))
        
        # The LLM's response is a string, we need to parse it as JSON
        output_dict = json.loads(bias_report.content)
        
        state['political_bias_output'] = json.dumps(output_dict)
        logger.info(f"  -> Political Bias Output: {state['political_bias_output']}")
    except Exception as e:
        logger.error(f"  -> Error in political_bias_agent: {e}")
        state['political_bias_output'] = json.dumps({"error": "Failed to run model", "details": str(e)})

    return state
    
def report_generator(state: State) -> State:
    """
    Synthesizes the results from all agents and generates a final report.
    This function processes the agent outputs before passing them to the LLM.
    """
    logger = logging.getLogger("report_generator")
    logger.info("---RUNNING FINAL LLM AGENT---")
    
    # Get the API key using our new function
    groq_api_key = get_groq_api_key()
    if not groq_api_key:
        logger.critical("GROQ_API_KEY is not set. Cannot proceed with LLM agent.")
        state['final_report'] = "Error: GROQ_API_KEY not found."
        return state
    
    try:
        llm = ChatGroq(model="llama3-8b-8192", temperature=0, groq_api_key=groq_api_key)
        
        raw_text = state['raw_text']
        retrieved_context = state['retrieved_context']
        
        # --- PRE-PROCESSING AGENT OUTPUTS IN PYTHON ---
        
        # Helper function to process outputs
        def process_output(json_str):
            try:
                data = json.loads(json_str)
                confidence = float(data.get("confidence_score", data.get("score", data.get("confidence"))))
                
                # Check for inconclusive results
                if confidence < 0.5:
                    result = "inconclusive"
                else:
                    # Get the primary result label
                    if "lie_detection" in data:
                        result = data["lie_detection"]
                    elif "emotion" in data:
                        result = data["emotion"]
                    elif "political_bias" in data:
                        result = data["political_bias"]
                    else:
                        result = "unknown"
                
                # Format confidence as a percentage
                formatted_confidence = f"{confidence:.2f}%"
                
                return f"{result} (Confidence: {formatted_confidence})"
            except (json.JSONDecodeError, ValueError, TypeError) as e:
                logger.error(f"Failed to process JSON output: {json_str}. Error: {e}")
                return "inconclusive (Error processing output)"

        # Process each agent's output using the helper function
        lie_detection_summary = process_output(state['lie_detection_output'])
        sentiment_summary = process_output(state['sentiment_output'])
        political_bias_summary = process_output(state['political_bias_output'])

        # --- UPDATED PROMPT FOR THE LLM ---
        # The prompt is now much simpler as the LLM doesn't need to do the logic.
        prompt_template = PromptTemplate.from_template(
            """You are an expert AI report generator. Your task is to analyze a social media post for bias and manipulation.
            
            Based on the following analysis results, write a final report.
            
            Original Post: {raw_text}
            
            Analysis Results:
            - Lie Detection: {lie_detection_summary}
            - Emotion/Sentiment: {sentiment_summary}
            - Political Bias: {political_bias_summary}
            
            Data from your Memory:
            {retrieved_context}
            
            Perform two tasks:
            1. **Generate a Final Score:** Provide a single numerical bias score from 1 to 10, where 1 is completely unbiased and 10 is highly biased and manipulative.
            2.**Generate a Result : ** Classify the post as 'unbiased', 'nuetral', 'biased', or 'manipulative' based on the final score.
            3. **Generate an Explanation:** Write a detailed, easy-to-understand explanation for the score. The explanation must directly reference the analysis results but dont not include the name of the agents individually but use words like System,Analyser,etc to address those agents . Note: The explanation should briefly mention in a single sentence that systems experience was leveraged to inform this analysis.
            
            Format your final output clearly, starting with the score, followed by the result in the next line and then followed by the explanation in the next line.
            
            Example Final Report:
            Score: 8/10
            Result: Biased
            Explanation: This post scored high for bias and manipulation. The political bias agent classified it as 'right-leaning' with a high confidence of 97.00%. The emotion agent identified 'anger' as the dominant emotion, suggesting a strong emotional appeal rather than a factual one. The lie detection agent was inconclusive on this specific claim. Historical data was leveraged to inform this analysis.
            """
        )
        
        prompt_with_data = prompt_template.format(
            raw_text=raw_text,
            lie_detection_summary=lie_detection_summary,
            sentiment_summary=sentiment_summary,
            political_bias_summary=political_bias_summary,
            retrieved_context=json.dumps(retrieved_context, indent=2)
        )
        
        final_report_message = llm.invoke(prompt_with_data)
        final_report_text = final_report_message.content
        
        state['final_report'] = final_report_text
        logger.info("Final Report generated successfully.")
        logger.info(f"Final Report:\n{state['final_report']}")
    except Exception as e:
        logger.critical(f"A critical error occurred in the report_generator: {e}")
        state['final_report'] = f"Error: Failed to generate report. Details: {str(e)}"
    
    return state


def memory_saver(state: State) -> State:
    """
    Saves the new post and its final report to the ChromaDB collection.
    This runs just before the graph ends.
    """
    logger = logging.getLogger("memory_saver")
    logger.info("---RUNNING PERSISTENCE AGENT---")

    raw_text = state.get('raw_text')
    final_report = state.get('final_report')
    
    if not raw_text or not final_report:
        logger.warning("Skipping persistence: raw_text or final_report is missing.")
        return state

    try:
        collection = setup_chromadb()
        if collection:
            unique_id = str(uuid.uuid4())
            collection.add(
                documents=[raw_text],
                embeddings=[get_embedding(raw_text)],
                metadatas=[{"final_report": final_report}],
                ids=[unique_id]
            )
            logger.info(f"Successfully saved new report to ChromaDB with ID: {unique_id}")
    except Exception as e:
        logger.error(f"Error in memory_saver: {e}")

    return state


# --- Graph Construction ---
def create_graph():
    """
    Creates and compiles the LangGraph state machine.
    This uses a fan-out/fan-in pattern to execute agents in parallel.
    """
    main_logger.info("Creating the LangGraph workflow.")
    workflow = StateGraph(State)
    
    # Add nodes to the graph
    workflow.add_node("context_retriever", context_retriever)
    workflow.add_node("analysis_engine", analysis_engine)
    workflow.add_node("report_generator", report_generator)
    workflow.add_node("memory_saver", memory_saver)

    # Set the START point
    workflow.set_entry_point("context_retriever")
        
    # Define the sequential flow
    workflow.add_edge("context_retriever", "analysis_engine")
    workflow.add_edge("analysis_engine", "report_generator")
    workflow.add_edge("report_generator", "memory_saver")
        
    # The memory_saver agent is the end of the workflow
    workflow.add_edge("memory_saver", END)

    app = workflow.compile()
    main_logger.info("LangGraph workflow compiled successfully.")
    return app


# --- Utility Function to Delete Memory ---
def delete_all_memory():
    """
    Deletes the entire ChromaDB collection used for memory.
    """
    main_logger.info("Attempting to delete the ChromaDB 'reports' collection.")
    try:
        client = chromadb.PersistentClient(path="memory")
        client.delete_collection(name="reports")
        print("\nSuccessfully deleted all memory from ChromaDB.")
        main_logger.info("Successfully deleted ChromaDB collection.")
    except Exception as e:
        print(f"\nCould not delete memory. It may not exist yet. Error: {e}")
        main_logger.warning(f"Failed to delete ChromaDB collection. Error: {e}")

def download_models(models_to_check: List[str]):
    """
    Checks if a list of Hugging Face models are downloaded locally. If not, it downloads them.
    """
    main_logger.info("---Checking and downloading models---")
    
    for model_name in models_to_check:
        try:
            main_logger.info(f"Checking for model: {model_name} locally...")
            
            # The SentenceTransformer model has a different loading method
            if "sentence-transformers" in model_name:
                SentenceTransformer(model_name, cache_folder=os.environ['HUGGINGFACE_HUB_CACHE'])
            else:
                # Use transformers pipeline to check for a text-classification model
                pipeline("text-classification", model=model_name, local_files_only=True)
                
            main_logger.info(f"Model '{model_name}' found locally. Skipping download.")
        except (OSError, FileNotFoundError) as e:
            main_logger.warning(f"Model '{model_name}' not found locally. Downloading now...")
            try:
                if "sentence-transformers" in model_name:
                    SentenceTransformer(model_name, cache_folder=os.environ['HUGGINGFACE_HUB_CACHE'])
                else:
                    pipeline("text-classification", model=model_name)
                main_logger.info(f"Download of '{model_name}' completed successfully.")
            except Exception as download_error:
                main_logger.error(f"Failed to download model '{model_name}'. Error: {download_error}")
        except Exception as e:
            main_logger.error(f"An unexpected error occurred while checking model '{model_name}': {e}")
    main_logger.info("---Model check complete---")

def run_analysis_system(app, user_input: str):
    """
    Runs the LangGraph analysis system with a given user input.
    """
    main_logger.info("Checking for models.")
    # Check and download models before starting the analysis
    models_to_download = [
        "google/electra-small-discriminator", 
        "j-hartmann/emotion-english-distilroberta-base",
        "sentence-transformers/all-MiniLM-L6-v2"
    ]
    download_models(models_to_download)
    main_logger.info("Models are ready. Starting analysis.")

    main_logger.info("Starting a new analysis job.")
    
    initial_state = {"raw_text": user_input}
    
    main_logger.info(f"--- ANALYZING POST: '{user_input}' ---")
    
    final_state = app.invoke(initial_state)

    main_logger.info("--- GRAPH EXECUTION COMPLETE ---")
    
    return final_state

# def visualize_graph_notebook(graph: StateGraph) -> None:
#     """
#     Generates and displays a PNG image of the LangGraph workflow
#     within a Jupyter Notebook.
#     """
#     try:
#         # Get the graph object
#         graph_object = graph.get_graph()
        
#         # Draw the graph as a PNG and display it
#         display(Image(graph_object.draw_mermaid_png()))
        
#     except Exception as e:
#         print(f"Failed to visualize the graph: {e}")
#         print("Please ensure you have the necessary dependencies for mermaid rendering installed.")

# --- Main Execution Loop ---
if __name__ == "__main__":
    
    # Check for the API key at the start of the script for CLI
    if not get_groq_api_key():
        main_logger.critical("CRITICAL: GROQ_API_KEY environment variable is not set. Please set it and try again.")
        print("\n" + "="*50)
        print("ERROR: GROQ API key is not configured.")
        print("Please set the GROQ_API_KEY environment variable to continue.")
        print("e.g., export GROQ_API_KEY='your-key'")
        print("="*50)
        exit(1)

    main_logger.info("Initializing the LangGraph agentic system.")
    
    app = create_graph()
    #visualize_graph_notebook(app)
    
    print("\n\nWelcome to the Social Media Post Analyzer!")
    print("Enter a social media post to analyze its potential for misinformation, bias, and sentiment.")
    print("The system will learn from each post. Type 'delete' to clear its memory.")
    print("Type 'exit' or 'quit' to close the program.")
    
    while True:
        try:
            user_post = input("\nEnter your social media post here: \n> ")
            if user_post.lower() in ['exit', 'quit']:
                print("Exiting program. Goodbye!")
                main_logger.info("User requested to exit. Shutting down.")
                break
            
            elif user_post.lower() == 'delete':
                delete_all_memory()
                continue

            if not user_post.strip():
                print("Input cannot be empty. Please enter a post to analyze.")
                continue
            
            result = run_analysis_system(app, user_post)
            
            print("\n--- ANALYSIS RESULTS ---")
            print(f"Final Report:\n{result['final_report']}")
            main_logger.info("Analysis completed successfully.")

        except Exception as e:
            main_logger.error(f"An unexpected error occurred in the main loop: {e}")
            print(f"An unexpected error occurred: {e}")
            print("Please try again or restart the program.")