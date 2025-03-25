import os
import streamlit as st
from dotenv import load_dotenv
from langchain.schema import HumanMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel
from langsmith import traceable
import traceback

# ✅ Load API keys securely
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")

os.environ["GROQ_API_KEY"] = GROQ_API_KEY
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = LANGSMITH_API_KEY

# ✅ Initialize LLM (Using Groq Llama3-8B)
llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama3-8b-8192")

# ✅ Define the LegalState Model
class LegalState(BaseModel):
    original_text: str
    tone: str
    complexity: int = 1
    rewritten_text: str = None
    summary: str = None
    key_clauses: str = None
    risk_analysis: str = None
    compliance_report: str = None
    contract_suggestions: str = None
    legal_arguments: str = None
    formatted_text: str = None
    comparison_result: str = None
    final_report: str = None

# ✅ Function to invoke LLM with error handling
def generate_response(prompt):
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content
    except Exception as e:
        return f"❌ Error: {str(e)}"

# ✅ Define Worker Functions
@traceable(name="Rewrite Legal Text")
def rewrite_text(state: LegalState):
    prompt = f"""Rewrite this legal text in '{state.tone}' tone with complexity level {state.complexity}:

    {state.original_text}"""
    return {"rewritten_text": generate_response(prompt)}

@traceable(name="Summarize Legal Text")
def summarize_text(state: LegalState):
    return {"summary": generate_response(f"Summarize this legal text:\n\n{state.rewritten_text}")}

@traceable(name="Extract Key Clauses")
def extract_clauses(state: LegalState):
    return {"key_clauses": generate_response(f"Extract key legal clauses:\n\n{state.rewritten_text}")}

@traceable(name="Detect Risks in Document")
def detect_risks(state: LegalState):
    return {"risk_analysis": generate_response(f"Analyze for risks:\n\n{state.rewritten_text}")}

@traceable(name="Check Compliance")
def check_compliance(state: LegalState):
    return {"compliance_report": generate_response(f"Check legal compliance:\n\n{state.rewritten_text}")}

@traceable(name="Suggest Contract Improvements")
def suggest_improvements(state: LegalState):
    return {"contract_suggestions": generate_response(f"Suggest improvements:\n\n{state.rewritten_text}")}

@traceable(name="Generate Legal Arguments")
def generate_arguments(state: LegalState):
    return {"legal_arguments": generate_response(f"Generate legal arguments:\n\n{state.rewritten_text}")}

@traceable(name="Format Legal Document")
def format_document(state: LegalState):
    return {"formatted_text": generate_response(f"Format this legal document:\n\n{state.rewritten_text}")}

@traceable(name="Compare Original vs Rewritten")
def compare_texts(state: LegalState):
    prompt = f"Compare original vs rewritten:\n\nOriginal: {state.original_text}\nRewritten: {state.rewritten_text}"
    return {"comparison_result": generate_response(prompt)}

# ✅ Build LangGraph Workflow
builder = StateGraph(LegalState)
builder.add_node("rewrite_text", rewrite_text)
builder.add_node("summarize_text", summarize_text)
builder.add_node("extract_clauses", extract_clauses)
builder.add_node("detect_risks", detect_risks)
builder.add_node("check_compliance", check_compliance)
builder.add_node("suggest_improvements", suggest_improvements)
builder.add_node("generate_arguments", generate_arguments)
builder.add_node("format_document", format_document)
builder.add_node("compare_texts", compare_texts)

builder.add_edge(START, "rewrite_text")
for node in ["summarize_text", "extract_clauses", "detect_risks", "check_compliance", "suggest_improvements", "generate_arguments", "format_document", "compare_texts"]:
    builder.add_edge("rewrite_text", node)

graph = builder.compile()

# ✅ Streamlit UI
def main():
    st.title("📜 AI-Powered Legal Text Processor")
    st.write("A smart legal document assistant powered by LLMs.")

    original_text = st.text_area("Enter Legal Text:")
    tone = st.radio("Select Tone:", ["Formal", "Empathetic", "Neutral", "Strength-Based"])
    
    if st.button("Process Text"):
        input_data = {"original_text": original_text, "tone": tone, "complexity": 1}
        result = graph.invoke(input_data)
        
        st.subheader("🔹 Rewritten Text")
        st.markdown(f"**{result['rewritten_text']}**")

        st.subheader("📌 Summary")
        st.write(result['summary'])
        
        st.subheader("📜 Key Clauses")
        st.write(result['key_clauses'])
        
        st.subheader("⚠️ Risk Analysis")
        st.write(result['risk_analysis'])
        
        st.subheader("✅ Compliance Report")
        st.write(result['compliance_report'])
        
        st.subheader("💡 Contract Suggestions")
        st.write(result['contract_suggestions'])
        
        st.subheader("⚖️ Legal Arguments")
        st.write(result['legal_arguments'])
        
        st.subheader("📝 Formatted Legal Document")
        st.write(result['formatted_text'])
        
        st.subheader("🔍 Comparison Result")
        st.write(result['comparison_result'])

if __name__ == "__main__":
    main()
