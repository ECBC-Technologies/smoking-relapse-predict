
import os
import streamlit as st
from PyPDF2 import PdfReader

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

#################################
# 1. Load environment variables
#################################
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_GEN_MODEL = os.getenv("OPENAI_GEN_MODEL", "gpt-4o-2024-05-13")
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.0"))

#################################
# 2. Streamlit Page Config & Custom CSS
#################################

st.set_page_config(
    page_title="Smoking Relapse Prediction Chatbot",
    page_icon="🩺",
    layout="wide"
)

# Hide default Streamlit elements for a cleaner look
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;} /* Hide the top-right hamburger menu */
footer {visibility: hidden;}     /* Hide the 'Powered by Streamlit' footer */
</style>
"""

# Custom CSS for a vertical sidebar menu with a professional look
vertical_menu_css = """
<style>
.sidebar-menu-container {
    margin-top: 1rem;
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
}

/* Each item is a button with our custom classes */
.sidebar-menu-button {
    display: block;
    width: 100%;
    text-align: left;
    font-size: 1rem;
    font-weight: 500;
    padding: 12px 16px;
    border: 1px solid #ddd;
    border-radius: 6px;
    background-color: #fff;
    color: #333;
    cursor: pointer;
    transition: background-color 0.3s ease, box-shadow 0.3s ease, border-left 0.3s ease;
}

.sidebar-menu-button:hover {
    background-color: #f1f1f1;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}

.sidebar-menu-button-active {
    background-color: #e3fce3;              /* subtle green background */
    border-left: 6px solid #4CAF50;         /* thick green left border */
    color: #4CAF50;
    font-weight: 600;
    box-shadow: inset 0 0 8px rgba(76, 175, 80, 0.1);
}
</style>
"""

st.markdown(hide_streamlit_style, unsafe_allow_html=True)
st.markdown(vertical_menu_css, unsafe_allow_html=True)


#################################
# 3. Helper Functions
#################################

def extract_text_from_pdf(pdf_file) -> str:
    """
    Extract all text from an uploaded PDF file using PyPDF2.
    Returns a single string with the PDF text.
    """
    reader = PdfReader(pdf_file)
    all_text = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            all_text.append(text)
    return "\n".join(all_text)

def chunk_text(text: str, chunk_size=1000, chunk_overlap=100) -> list:
    """
    Break a long text into smaller chunks for better embedding and retrieval.
    Using RecursiveCharacterTextSplitter from LangChain for more robust splitting.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", "!", "?", ",", ""]
    )
    return splitter.split_text(text)

def create_vectorstore(docs: list) -> FAISS:
    """
    Given a list of Document objects, create a FAISS vectorstore using
    OpenAIEmbeddings for text embeddings.
    """
    if not OPENAI_API_KEY:
        raise ValueError("OpenAI API key is missing. Please set OPENAI_API_KEY.")
    
    embeddings = OpenAIEmbeddings(
        openai_api_key=OPENAI_API_KEY,
        model=OPENAI_EMBED_MODEL
    )
    vectorstore = FAISS.from_documents(docs, embedding=embeddings)
    return vectorstore

def create_qa_chain(vectorstore: FAISS) -> RetrievalQA:
    """
    Create a RetrievalQA chain that uses ChatOpenAI as the LLM
    and the vectorstore as the retriever.
    """
    if not OPENAI_API_KEY:
        raise ValueError("OpenAI API key is missing. Please set OPENAI_API_KEY.")

    FORMAT_INSTRUCTIONS = """
        You are a public health researcher specializing in smoking cessation.
        Given the following extracted segments from research PDFs (context)
        and the user's question, please respond with the following format and add 
        any other relevant factors you can find in the documents:

        Based on the provided reserach paper, several factors increase the likelihood of smoking relapse, including:

        Age Factor:
        [Explain findings from context regarding age]

        Living with Other Smokers:
        [Explain findings from context regarding living situation]

        Lifestyle Factors (Partying) or other relevant factors:
        [Explain lifestyle or relevant contextual factors]

        Estimating <Name>'s Risk:
        [Provide a reasoned estimate for the user’s risk based on the context. 
        Include numeric figures or ranges if the context references them.]

        Conclusion:
        [Summarize the final stance with disclaimers if needed]

        Context: {context}

        User Question: {question}
    """
    # Create a PromptTemplate
    custom_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=FORMAT_INSTRUCTIONS
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # Create the chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(
            openai_api_key=OPENAI_API_KEY,
            model_name=OPENAI_GEN_MODEL,
            temperature=OPENAI_TEMPERATURE
        ),
        chain_type="stuff",  # "stuff" just concatenates the context + prompt
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={
            "prompt": custom_prompt
        }
    )

    return qa_chain


#################################
# 4. Vertical Menu for Navigation
#################################

def vertical_menu(menu_items):
    """
    Renders a vertical clickable menu in the sidebar using st.button for each item,
    styled to look more professional. Stores the selected item in st.session_state["selected_menu"].
    Returns the currently selected item.
    """
    # Container for our custom-styled menu
    st.sidebar.markdown('<div class="sidebar-menu-container">', unsafe_allow_html=True)

    # Ensure we have a default selection
    if "selected_menu" not in st.session_state:
        st.session_state["selected_menu"] = menu_items[0]

    for item in menu_items:
        # Determine if this item is active
        is_active = (item == st.session_state["selected_menu"])
        button_class = "sidebar-menu-button-active" if is_active else "sidebar-menu-button"

        # We'll create a clickable button with a custom class for styling
        # Trick: We use st.markdown(...) with a form and an <input type="submit"> for advanced styling, 
        # but we'll keep it simpler with st.button to avoid complexity.
        clicked = st.sidebar.button(
            item,
            key=f"vertical_menu_{item}",
            help=f"Go to {item}",
        )

        # After the button is rendered, we inject a little HTML to apply the custom class via style
        # This is a hack because st.button doesn't let us set a custom HTML class directly.
        # We'll do an invisible text element that matches the same key to identify it in the DOM.
        st.sidebar.markdown(
            f"""
            <style>
            /* This targets the div containing the button with the matching key */
            div [key="vertical_menu_{item}"] {{
                margin: 0 !important;
            }}
            div [key="vertical_menu_{item}"] > button:first-child {{
                all: unset !important;
                display: block;
                width: 100%;
                text-align: left;
                font-size: 1rem;
                font-weight: 500;
                padding: 12px 16px;
                margin-bottom: 8px;
                border: 1px solid #ddd;
                border-radius: 6px;
                background-color: #fff;
                color: #333;
                cursor: pointer;
                transition: background-color 0.3s ease, box-shadow 0.3s ease, border-left 0.3s ease;
            }}
            div [key="vertical_menu_{item}"] > button:first-child:hover {{
                background-color: #f1f1f1;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            }}
            div [key="vertical_menu_{item}"] > button:first-child:focus {{
                outline: none;
            }}
            div [key="vertical_menu_{item}"] > button.sidebar-menu-button-active {{
                background-color: #e3fce3;
                border-left: 6px solid #4CAF50;
                color: #4CAF50;
                font-weight: 600;
                box-shadow: inset 0 0 8px rgba(76, 175, 80, 0.1);
            }}
            </style>
            """,
            unsafe_allow_html=True,
        )

        # If user clicked, update the selected menu item
        if clicked:
            st.session_state["selected_menu"] = item

        # If this is the active item, we replace the button's default style with our "active" class
        if is_active:
            # We can't directly change st.button after it's rendered, so we rely on the CSS we injected
            # to match the item and key, applying the "sidebar-menu-button-active" style. 
            # But let's also rename the button to the active class with a small hack:
            st.sidebar.write(
                f"""
                <script>
                var buttons = window.parent.document.querySelectorAll('div[key="vertical_menu_{item}"] button');
                if (buttons.length > 0) {{
                    buttons[0].classList.add("sidebar-menu-button-active");
                }}
                </script>
                """,
                unsafe_allow_html=True
            )

    st.sidebar.markdown('</div>', unsafe_allow_html=True)
    return st.session_state["selected_menu"]


#################################
# 5. Main Streamlit App
#################################

def main():
    # Header
    st.markdown("<h1 style='text-align: center;'>AI Chatbot for Smoking Relapse Prediction</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: gray;'>Upload research PDFs, then ask questions about smoking relapse.</p>", unsafe_allow_html=True)
    st.divider()

    # Ensure API key is available
    if not OPENAI_API_KEY:
        st.error("OpenAI API key not found. Please set OPENAI_API_KEY in your environment.")
        return

    # Session state for vectorstore and QA chain
    if "vectorstore" not in st.session_state:
        st.session_state["vectorstore"] = None
    if "qa_chain" not in st.session_state:
        st.session_state["qa_chain"] = None

    # Menu items
    menu_items = ["Train Model", "Predict"]
    selected = vertical_menu(menu_items)

    if selected == "Train Model":
        ######################################
        # TRAIN MODEL SECTION
        ######################################
        st.subheader("Train Model")
        st.write("Upload your research PDFs below. The system will extract and embed the text for Q&A.")

        uploaded_files = st.file_uploader(
            "Upload one or more Research PDFs",
            type=["pdf"],
            accept_multiple_files=True
        )

        if st.button("Train Model"):
            if not uploaded_files:
                st.warning("Please upload at least one PDF.")
            else:
                st.info("Reading and indexing PDFs...")
                all_docs = []
                for pdf_file in uploaded_files:
                    # Extract text
                    raw_text = extract_text_from_pdf(pdf_file)
                    # Chunk text
                    chunks = chunk_text(raw_text)
                    # Convert to Document objects
                    docs = [Document(page_content=chunk) for chunk in chunks]
                    all_docs.extend(docs)

                # Create vectorstore
                try:
                    vectorstore = create_vectorstore(all_docs)
                    qa_chain = create_qa_chain(vectorstore)

                    # Save in session state
                    st.session_state["vectorstore"] = vectorstore
                    st.session_state["qa_chain"] = qa_chain
                    st.success("Model trained successfully! Switch to 'Predict' to query the data.")
                except Exception as e:
                    st.error(f"Error creating vector store: {e}")

    elif selected == "Predict":
        ######################################
        # PREDICT / CHATBOT SECTION
        ######################################
        st.subheader("Prediction")
        st.write("Ask a question about smoking relapse.")

        user_query = st.text_area(
            "Enter your query:",
            value="John is 23 years, smoked for 3 years and quit 3m onths ago. His job is stressfull and he likes to party. What is the probability he'll relapse?"
        )

        if st.button("Predict Now"):
            qa_chain = st.session_state.get("qa_chain", None)
            if not qa_chain:
                st.warning("No model found. Please train the model first (in 'Train Model').")
            else:
                with st.spinner("Generating answer..."):
                    try:
                        result = qa_chain(user_query)
                        answer = result["result"]
                        source_docs = result["source_documents"]

                        st.markdown("### Answer")
                        st.write(answer)

                        st.markdown("### Source Documents")
                        for i, doc in enumerate(source_docs, start=1):
                            st.write(f"**Source {i}**: {doc.page_content[:200]}...")
                    except Exception as e:
                        st.error(f"Error during prediction: {e}")


if __name__ == "__main__":
    main()
