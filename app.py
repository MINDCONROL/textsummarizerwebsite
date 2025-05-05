import streamlit as st
# hide_streamlit_style = """
#             <style>
#             #MainMenu {visibility: hidden;}
#             footer {visibility: hidden;}
#             header {visibility: hidden;}
#             </style>
#             """
# st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# st.title("HunterAI") # You can still set a title that appears in the app content

from bs4 import BeautifulSoup
from transformers import pipeline, T5Tokenizer, T5ForConditionalGeneration


from transformers import pipeline
import io
from PyPDF2 import PdfReader
from docx import Document
from pptx import Presentation
import streamlit.components.v1 as components
import os
import tempfile
from typing import Optional
from transformers import T5Tokenizer, T5ForConditionalGeneration
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.probability import FreqDist
import string
import re

#from streamlit_option_menu import option_menu
choice = st.sidebar.selectbox("Select an option", ["Home","Document","URL","About"])
if choice == "Home":
    st.title("Welcome to the Home Page")
    st.write("This is the home page of the Streamlit app.")
    st.write("You can use this app to summarize text using the T5 model.")
    st.write("Please enter the text you want to summarize in the text area below.")
    text = st.text_area("Enter text to summarize", height=300)
    if st.button("Summarize"):
        from transformers import pipeline
        summarizer = pipeline("summarization", model="t5-base", tokenizer="t5-base", framework="pt")
        summary = summarizer(text, max_length=100, min_length=30, do_sample=False)
        st.write("Summary:")
        st.write(summary[0]['summary_text'])

        

elif choice == "Document":
    ## Required NLTK downloads
    nltk.download('punkt')
    nltk.download('stopwords')

# File processing libraries
    from PyPDF2 import PdfReader
    from docx import Document
    from pptx import Presentation

# Initialize T5 model for summarization
@st.cache_resource
def load_summarization_model():
    try:
        model_name = "t5-small"
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        model = T5ForConditionalGeneration.from_pretrained(model_name)
        return tokenizer, model
    
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        st.stop()

tokenizer, model = load_summarization_model()

def extract_text_from_file(file_path: str, file_type: str) -> Optional[str]:
    """Extract text from different file types"""
    try:
        if file_type == "application/pdf":
            reader = PdfReader(file_path)
            text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
            return text
        
        elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = Document(file_path)
            text = "\n".join([para.text for para in doc.paragraphs if para.text])
            return text
        
        elif file_type == "application/vnd.openxmlformats-officedocument.presentationml.presentation":
            prs = Presentation(file_path)
            text = []
            for slide in prs.slides:
                for shape in slide.shapes:
                    if hasattr(shape, "text"):
                        text.append(shape.text)
            return "\n".join(text)
        
        elif file_type == "text/plain":
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        
        else:
            st.error(f"Unsupported file type: {file_type}")
            return None
    
    except Exception as e:
        st.error(f"Error extracting text: {str(e)}")
        return None

def preprocess_text(text: str) -> str:
    """Clean and preprocess text for summarization"""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove special characters except basic punctuation
    text = re.sub(r'[^\w\s.,;:!?()-]', '', text)
    return text

def generate_t5_summary(text: str, max_length: int = 512) -> str:
    """Generate abstractive summary using T5 model"""
    try:
        # Preprocess and truncate text to model's max input length
        input_text = "summarize: " + text[:4000]  # T5-small has limit of 512 tokens
        
        inputs = tokenizer.encode(
            input_text,
            return_tensors="pt",
            max_length=512,
            truncation=True
        )
        
        summary_ids = model.generate(
            inputs,
            max_length=max_length,
            min_length=30,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True
        )
        
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary
    
    except Exception as e:
        st.error(f"Error in T5 summarization: {str(e)}")
        return "Summary generation failed."

def extract_key_phrases(text: str, num_phrases: int = 5) -> list:
    """Extract key phrases using NLTK"""
    # Tokenize and clean
    words = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english') + list(string.punctuation))
    words = [word for word in words if word not in stop_words and word.isalnum()]
    
    # Calculate frequency distribution
    freq_dist = FreqDist(words)
    return [item[0] for item in freq_dist.most_common(num_phrases)]

def categorize_content(text: str) -> dict:
    """Categorize content into sections"""
    sentences = sent_tokenize(text)
    num_sentences = len(sentences)
    
    # Heuristic for dividing content
    if num_sentences < 5:
        return {
            "Heading": "Main Content",
            "Subheadings": [],
            "Points": sentences,
            "Summary": " ".join(sentences[:2])
        }
    
    # Divide into sections
    heading = sentences[0]
    subheadings = []
    points = []
    
    # First quarter for heading/subheading
    quarter = max(1, num_sentences // 4)
    heading = sentences[0]
    if num_sentences > 3:
        subheadings.append(sentences[1])
    
    # Middle for points
    points = sentences[quarter:-quarter]
    
    # Last part for summary
    summary_sents = sentences[-quarter:]
    
    return {
        "Heading": heading,
        "Subheadings": subheadings,
        "Points": points,
        "Summary": " ".join(summary_sents)
    }

def main():
    st.title("Document Text Extraction and Summarization")
    st.write("Upload a document (PDF, DOCX, PPTX, or TXT) to extract text and generate a summary")
    
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["pdf", "docx", "pptx", "txt"]
    )
    
    if uploaded_file is not None:
        # Create a temporary file with proper extension
        file_ext = os.path.splitext(uploaded_file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            file_path = tmp_file.name
        
        try:
            # Extract text
            file_type = uploaded_file.type
            raw_text = extract_text_from_file(file_path, file_type)
            
            if raw_text:
                # Clean text
                cleaned_text = preprocess_text(raw_text)
                
                # Display original text
                with st.expander("View Extracted Text"):
                    st.text_area("Extracted Text", cleaned_text, height=300)
                
                # Process and summarize
                st.subheader("Document Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### Key Information")
                    key_phrases = extract_key_phrases(cleaned_text)
                    st.write("**Key Phrases:**")
                    for phrase in key_phrases:
                        st.write(f"- {phrase}")
                    
                    categorized = categorize_content(cleaned_text)
                    st.write("**Main Heading:**")
                    st.info(categorized["Heading"])
                    
                    if categorized["Subheadings"]:
                        st.write("**Subheadings:**")
                        for sub in categorized["Subheadings"]:
                            st.success(f"- {sub}")
                
                with col2:
                    st.markdown("### Abstractive Summary (T5)")
                    t5_summary = generate_t5_summary(cleaned_text)
                    st.write(t5_summary)
                    
                    st.markdown("### Quick Summary Points")
                    if categorized["Points"]:
                        for point in categorized["Points"][:5]:  # Limit to 5 points
                            st.write(f"- {point}")
                
                st.markdown("### Conclusion")
                st.write(categorized["Summary"])
        
        finally:
            # Clean up the temporary file
            try:
                os.unlink(file_path)
            except:
                pass

if __name__ == "__main__":
    main()
elif choice == "URL":
      import requests
      from bs4 import BeautifulSoup
      from transformers import pipeline

      st.title("URL Summarizer")

      url = st.text_input("Enter URL to summarize:")

      if st.button("Summarize URL"):
          if url:
              try:
                  response = requests.get(url)
                  response.raise_for_status()  # Raise an exception for bad status codes
                  soup = BeautifulSoup(response.content, 'html.parser')
                  text = soup.get_text()

                  # Initialize the summarizer outside the button's conditional block
                  summarizer = pipeline("summarization", model="t5-base", tokenizer="t5-base", framework="pt")

                  summary = summarizer(text, min_length=100, do_sample=False)
                  st.subheader("Summary:")
                  st.write(summary[0]['summary_text'])

              except requests.exceptions.RequestException as e:
                  st.error(f"Error fetching URL: {e}")
              except Exception as e:
                  st.error(f"An error occurred during summarization: {e}")
          else:
              st.warning("Please enter a URL.")
        
          # st.write("The summarization process uses the T5 model, which is a transformer-based model pre-trained on a large corpus of text.")
          # st.write("The model is capable of generating high-quality summaries for a wide range of text inputs.")
          # st.write("The summarization process may take some time depending on the length of the text and the complexity of the content.")
          # st.write("Please ensure that the URL you enter is accessible and contains text content that can be summarized.")
          # st.write("If the URL is not accessible or does not contain text content, the summarization process may fail.")
          # st.write("You can adjust the maximum and minimum length of the summary by modifying the parameters in the code.")


elif choice == "About":
    st.title("About this App")
    st.write("This app uses the T5 model from Hugging Face's Transformers library to summarize text.")
    st.write("The T5 model is a transformer-based model that has been pre-trained on a large corpus of text.")
    st.write("You can enter any text in the text area and click the 'Summarize' button to get a summary.")
    st.write("This app is built using Streamlit, a Python library for creating web apps.")

    st.write("You can find the source code for this app on GitHub.")
    st.write("If you have any questions or feedback, please feel free to reach out.")
    st.write("Thank you for using this app!")


