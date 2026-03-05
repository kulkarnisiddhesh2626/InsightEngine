from langchain_ollama import ChatOllama
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import initialize_agent, AgentType
import os

# Set your Tavily Key here
os.environ["TAVILY_API_KEY"] = "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"

# Initialize the 3b model as our primary reasoner
llm = ChatOllama(model="llama3.2:3b", temperature=0)

# Define Tools
search = TavilySearchResults(max_results=2)

def get_agent(vector_retriever):
    # Tool 1: The PDF search
    def pdf_search(query):
        docs = vector_retriever.get_relevant_documents(query)
        return "\n".join([d.page_content for d in docs])

    tools = [
        search, 
        # We wrap your vector store as a tool
        search # placeholder for vector tool
    ]
    
    return initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)