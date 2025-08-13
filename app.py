import streamlit as st
import os
import sys

# Add the parent directory to the path to import agent.py
# This is necessary if agent.py is not directly in the same directory as app.py
# Adjust this path if your directory structure is different
# For example, if agent.py is in a 'backend' folder, it might be:
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend')))
# Assuming agent.py is in the same directory for simplicity as per previous context
sys.path.append(os.path.abspath(os.path.dirname(__file__)))


from agent import create_graph, run_analysis_system

# --- Streamlit UI ---
st.set_page_config(layout="centered", page_title="Vote Gaurd")

st.title("Vote Gaurd")
st.markdown("Enter a social media post below to analyze its potential for misinformation, bias, and sentiment.")

# Initialize the graph once and cache it
@st.cache_resource
def get_graph():
    # Ensure environment variables are loaded for the cached resource
    from dotenv import load_dotenv
    load_dotenv()
    # Set Hugging Face cache directory for models
    os.environ['HUGGINGFACE_HUB_CACHE'] = './models'
    return create_graph()

app = get_graph()

# Text input for the user's post
user_post = st.text_area("Paste your social media post here:", height=150, placeholder="Type your social media post here...")

# Analyze Post button
if st.button("Analyze Post", use_container_width=True, type="primary"):
    if user_post.strip() == "":
        st.error("Please enter a post to analyze.")
    else:
        with st.spinner("Analyzing post... This may take a moment as models are loaded and processing occurs."):
            try:
                final_state = run_analysis_system(app, user_post)
                report = final_state.get('final_report', 'Analysis failed to produce a report.')
                
                st.subheader("Report") # Changed heading from "Final Report" to "Report"
                lines = report.split('\n')
                lines.pop(0)  # Remove the first line which is just a header    
                for line in lines:
                    if line.strip().startswith("Score:"):
                        # Display the score in a prominent way, maybe with a metric
                        pass
                    elif line.strip().startswith("Result:"):
                        # Display the result as a heading
                        st.markdown(f"**Result :** {line.split(': ')[1]}")
                    elif line.strip().startswith("Explanation:"):
                        # Display the explanation as plain text
                        st.markdown(f"**Explanation :** {line.split('Explanation:')[1].strip()}")
                    else:
                        # For any other text, just write it
                        st.write(line) 
            except Exception as e:
                st.error(f"An error occurred during analysis: {e}")
                st.info("Please check your API keys and ensure all models have downloaded correctly.")


# Clear Memory button - Centered
# if st.button("Clear Agent's Memory", use_container_width=True):
#     delete_all_memory()
#     st.success("Agent's memory has been successfully cleared.")

st.markdown("---") # Separator

# # Info for running the app
# st.sidebar.header("How to Run")
# st.sidebar.info("1. Ensure you have a `.env` file in the same directory with your `GROQ_API_KEY` set.\n"
#                 "2. Make sure all required Python packages are installed (`pip install -r requirements.txt`).\n"
#                 "3. Navigate to your project directory in the terminal.\n"
#                 "4. Run: `streamlit run app.py`")
# st.sidebar.header("Notes")
# st.sidebar.info("The first run might take longer as ML models are downloaded and cached to the `./models` folder.")

