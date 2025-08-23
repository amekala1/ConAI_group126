import streamlit as st

# set_page_config MUST be the first Streamlit command
st.set_page_config(
    page_title="💸 Financial QA Chatbot",
    page_icon="💬",
    layout="wide",
)

# Now import other modules
import rag_full_system
import os
import time
import preprocessing

# Load components and caching
@st.cache_resource
def load_rag_components():
    try:
        return rag_full_system.load_all_components()
    except Exception as e:
        st.error(f"❌ Failed to load RAG components. Error: {e}")
        return None

# Initialize preprocessing and load components
initialize = preprocessing.initialize()  # Ensure preprocessing is done before loading components
rag_components = load_rag_components()
#ft_components = load_ft_components()

if rag_components is None:
    st.stop()

# Add a bit of custom CSS
st.markdown("""
    <style>
    .stChatMessage.user {background-color: #e1f5fe; border-radius: 10px; padding: 10px;}
    .stChatMessage.assistant {background-color: #f1f8e9; border-radius: 10px; padding: 10px;}
    .metric-card {
        background: #ffffff;
        padding: 15px;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("💸 Financial QA Chatbot")
st.markdown("Ask financial questions about Apple's 2023/2024 performance. Compare **RAG vs Fine-Tuned Model**.")

# Sidebar controls
st.sidebar.header("⚙️ System Settings")
# Only RAG System is available
model_choice = 'RAG System'

# =================================================================================================
# Chat History
# =================================================================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# =================================================================================================
# User Input
# =================================================================================================
if prompt := st.chat_input("Ask a question about Apple's 2023/2024 financials..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""


        # Only RAG System is available
        with st.spinner("🔎 Searching for an answer with the RAG System..."):
            result = rag_full_system.run_rag_system(prompt, rag_components)

            # Extract results
            answer = result['answer']
            confidence = result['retrieval_confidence']
            time_taken = result['response_time']
            guardrail_message = result['guardrail_message']

            # Display metrics as cards
            col1, col2, col3, col4 = st.columns(4)
            with col1: st.metric("Method", "RAG System")
            with col2: st.metric("Confidence", f"{confidence:.2f}")
            with col3: st.metric("Time (s)", f"{time_taken:.2f}")
            with col4: st.metric("Guardrail", guardrail_message)

            st.markdown(f"### 💡 Answer\n{answer}")

            # Retrieved passages
            with st.expander("📑 Show Retrieved Passages"):
                if result['retrieved_passages']:
                    for i, (passage, metadata) in enumerate(zip(result['retrieved_passages'], result['retrieved_metadata'])):
                        st.markdown(f"**Source:** `{metadata['source']}`")
                        st.write(passage)
                else:
                    st.info("No relevant passages were retrieved.")

            full_response = answer

        message_placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})