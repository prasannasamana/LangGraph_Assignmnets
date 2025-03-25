import os
import streamlit as st
import traceback
from langgraph.graph import StateGraph, START, END
from langchain.schema import HumanMessage
from langchain_groq import ChatGroq
from langsmith import traceable
from typing import TypedDict

# Load API Keys (Set in Hugging Face Spaces)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")

# Ensure API Keys are set
if not GROQ_API_KEY or not LANGSMITH_API_KEY:
    st.error("⚠️ Please set GROQ_API_KEY and LANGSMITH_API_KEY in your environment variables.")
    st.stop()

# Initialize Groq LLM
llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama3-8b-8192")

# Define State
class State(TypedDict):
    code_snippet: str
    review_comments: str
    suggestions: str
    documentation: str
    test_cases: str

# Function to review the code
@traceable(name="Code Review")
def code_review(data):
    code_snippet = data.get("code_snippet", "")
    prompt = f"Review the following code and provide feedback:\n\n{code_snippet}"
    response = llm([HumanMessage(content=prompt)])
    return {"review_comments": response.content}

# Function to generate improvement suggestions
@traceable(name="Improvement Suggestions")
def improvement_suggestions(data):
    review_comments = data.get("review_comments", "")
    prompt = f"Based on this review feedback, suggest improvements:\n\n{review_comments}"
    response = llm([HumanMessage(content=prompt)])
    return {"suggestions": response.content}

# Function to generate documentation
@traceable(name="Code Documentation Generator")
def generate_documentation(data):
    code_snippet = data.get("code_snippet", "")
    prompt = f"Generate proper docstrings and inline comments for the following code:\n\n{code_snippet}"
    response = llm([HumanMessage(content=prompt)])
    return {"documentation": response.content}

# Function to generate test cases
@traceable(name="Test Case Suggestions")
def generate_test_cases(data):
    code_snippet = data.get("code_snippet", "")
    prompt = f"Based on the given code, generate appropriate unit test cases:\n\n{code_snippet}"
    response = llm([HumanMessage(content=prompt)])
    return {"test_cases": response.content}

# Create LangGraph Workflow
def make_code_review_graph():
    """Create a LangGraph workflow for automated code reviews"""
    graph_workflow = StateGraph(State)

    graph_workflow.add_node("code_review", code_review)
    graph_workflow.add_node("improvement_suggestions", improvement_suggestions)
    graph_workflow.add_node("generate_documentation", generate_documentation)
    graph_workflow.add_node("generate_test_cases", generate_test_cases)

    graph_workflow.add_edge(START, "code_review")
    graph_workflow.add_edge("code_review", "improvement_suggestions")
    graph_workflow.add_edge("improvement_suggestions", "generate_documentation")
    graph_workflow.add_edge("generate_documentation", "generate_test_cases")
    graph_workflow.add_edge("generate_test_cases", END)

    return graph_workflow.compile()

# Streamlit UI
st.title("🛠 AI-Powered Code Review with LangGraph & LangSmith")
st.write("Analyze and improve your code using AI-based feedback, suggestions, documentation, and test cases.")

# Input Field
code_snippet = st.text_area("📌 Paste your code snippet below:", height=200)

if st.button("🔍 Review Code"):
    if not code_snippet.strip():
        st.warning("⚠️ Please enter a valid code snippet.")
    else:
        try:
            review_agent = make_code_review_graph()
            result = review_agent.invoke({"code_snippet": code_snippet})

            # Display Results with Clear Formatting
            st.subheader("💡 Review Comments")
            st.write(result["review_comments"])

            st.subheader("🔧 Suggested Improvements")
            st.write(result["suggestions"])

            st.subheader("📖 Generated Documentation")
            st.code(result["documentation"], language="python")

            st.subheader("🧪 Suggested Test Cases")
            st.code(result["test_cases"], language="python")

        except Exception as e:
            st.error(f"⚠️ Error: {str(e)}")
            st.text(traceback.format_exc())
