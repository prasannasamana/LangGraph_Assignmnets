import os
import gradio as gr
import traceback
import torch
from langgraph.graph import StateGraph, START, END
from langchain.schema import HumanMessage
from langchain_groq import ChatGroq
from langsmith import traceable  # ✅ Added LangSmith for Debugging
from typing import TypedDict
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# ✅ Load API keys from Hugging Face Secrets
GROQ_API_KEY = os.getenv("GROQ_API_KEY")  
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")

# ✅ Set LangSmith Debugging
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = LANGSMITH_API_KEY

# ✅ Initialize Groq LLM (for content generation)
llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama3-8b-8192")

# ✅ Define State for LangGraph
class State(TypedDict):
    topic: str
    titles: list
    selected_title: str
    content: str
    summary: str
    translated_content: str
    tone: str
    language: str

# ✅ Function to generate multiple blog titles using Groq
@traceable(name="Generate Titles")  # ✅ Debugging with LangSmith
def generate_titles(data):
    topic = data.get("topic", "")
    prompt = f"Generate three short and catchy blog titles for the topic: {topic}. Each title should be under 10 words. Separate them with new lines."
    
    response = llm([HumanMessage(content=prompt)])
    titles = response.content.strip().split("\n")  
    
    return {"titles": titles, "selected_title": titles[0]}  

# ✅ Function to generate blog content with tone using Groq
@traceable(name="Generate Content")  # ✅ Debugging with LangSmith
def generate_content(data):
    title = data.get("selected_title", "")
    tone = data.get("tone", "Neutral")
    prompt = f"Write a detailed and engaging blog post in a {tone} tone based on the title: {title}"
    
    response = llm([HumanMessage(content=prompt)])
    return {"content": response.content.strip()}

# ✅ Function to generate summary using Groq
@traceable(name="Generate Summary")  # ✅ Debugging with LangSmith
def generate_summary(data):
    content = data.get("content", "")
    prompt = f"Summarize this blog post in a short and engaging way: {content}"
    
    response = llm([HumanMessage(content=prompt)])
    return {"summary": response.content.strip()}

# ✅ Load translation model (NLLB-200)
def load_translation_model():
    model_name = "facebook/nllb-200-distilled-600M"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_translation_model()

# ✅ Language codes for NLLB-200
language_codes = {
    "English": "eng_Latn",
    "Hindi": "hin_Deva",
    "Telugu": "tel_Telu",
    "Spanish": "spa_Latn",
    "French": "fra_Latn"
}

# ✅ Function to translate blog content using NLLB-200
@traceable(name="Translate Content")  # ✅ Debugging with LangSmith
def translate_content(data):
    content = data.get("content", "")
    language = data.get("language", "English")

    if language == "English":
        return {"translated_content": content}

    tgt_lang = language_codes.get(language, "eng_Latn")  

    # ✅ Split content into smaller chunks (Avoids token limit issues)
    max_length = 512  
    sentences = content.split(". ")  
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) < max_length:
            current_chunk += sentence + ". "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "

    if current_chunk:
        chunks.append(current_chunk.strip())

    # ✅ Translate each chunk separately and combine results
    translated_chunks = []
    for chunk in chunks:
        inputs = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True)
        translated_tokens = model.generate(**inputs, forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang))
        translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
        translated_chunks.append(translated_text.strip())

    full_translation = " ".join(translated_chunks)

    return {"translated_content": full_translation}

# ✅ Create LangGraph Workflow
def make_blog_generation_graph():
    """Create a LangGraph workflow for Blog Generation"""
    graph_workflow = StateGraph(State)

    # Define Nodes
    graph_workflow.add_node("title_generation", generate_titles)
    graph_workflow.add_node("content_generation", generate_content)
    graph_workflow.add_node("summary_generation", generate_summary)
    graph_workflow.add_node("translation", translate_content)  

    # Define Execution Order
    graph_workflow.add_edge(START, "title_generation")
    graph_workflow.add_edge("title_generation", "content_generation")
    graph_workflow.add_edge("content_generation", "summary_generation")
    graph_workflow.add_edge("content_generation", "translation")
    graph_workflow.add_edge("summary_generation", END)
    graph_workflow.add_edge("translation", END)

    return graph_workflow.compile()

# ✅ Function to generate blog content (Fixed)
def generate_blog(topic, tone, language):
    try:
        if not topic:
            return "⚠️ Please enter a topic.", "", "", "", ""

        blog_agent = make_blog_generation_graph()
        result = blog_agent.invoke({"topic": topic, "tone": tone, "language": language})

        return result["titles"], result["selected_title"], result["content"], result["summary"], result["translated_content"]

    except Exception as e:
        error_message = f"⚠️ Error: {str(e)}\n{traceback.format_exc()}"
        return error_message, "", "", "", ""

# ✅ Gradio UI
with gr.Blocks() as app:
    gr.Markdown(
        """
        ### 🌍 Why Translate?  
        - 🗣️ **Multilingual Support**  
        - 🌎 **Expand Reach**  
        - ✅ **Better Understanding**  
        - 🤖 **AI-Powered Accuracy**  
        """
    )

    gr.Interface(
        fn=generate_blog,
        inputs=[
            gr.Textbox(label="Enter a topic for your blog"),
            gr.Dropdown(["Neutral", "Formal", "Casual", "Persuasive", "Humorous"], label="Select Blog Tone", value="Neutral"),
            gr.Dropdown(["English", "Hindi", "Telugu", "Spanish", "French"], label="Translate Blog To", value="English"),
        ],
        outputs=[
            gr.Textbox(label="Suggested Blog Titles"),
            gr.Textbox(label="Selected Blog Title"),
            gr.Textbox(label="Generated Blog Content"),
            gr.Textbox(label="Blog Summary"),
            gr.Textbox(label="Translated Blog Content"),
        ],
        title="🚀 AI-Powered Blog Generator",
    )

# ✅ Launch the Gradio App
app.launch(share=True)

