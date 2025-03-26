# set API keys for Groq, Open AI and Tavil
import os
from dotenv import load_dotenv
load_dotenv

GROQ_API_KEY=os.environ.get("GROQ_API_KEY")
TAVILY_API_KEY=os.environ.get("TAVILY_API_KEY")
OPENAI_API_KEY=os.environ.get("OPENAI_API_KEY")
# Phase 1 complete

# set up LLMs and Tools
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults

openai_llm = ChatOpenAI(model='gpt-4o-mini')
groq_llm = ChatGroq(model='llama-3.3-70b-versatile')

# Phase 2 complete

# set up AI agent with search tool functionality
from langgraph.prebuilt import create_react_agent
# to get the response we want/filter i.e omit the meta data etc.
from langchain_core.messages.ai import AIMessage

def get_response_from_ai_agent(model_name,query,allow_search, system_prompt, provider):
    if provider=="Groq":
        model = ChatGroq(model=model_name)
    elif provider=="OpenAI":
        model = ChatOpenAI(model=model_name)

    tools = [TavilySearchResults(max_results=2)] if allow_search else []
    
    
    agent = create_react_agent( #inbuilt function create_react_agent
        model= model,
        tools= tools,
        # what kind of a role the agent will have
        state_modifier=system_prompt
    )

    # testing querries
    query= query
    state={"messages": query}
    # it will not only have the answer returend but the api calls and other meta data
    response = agent.invoke(state)
    # to get the AI message only
    messages= response.get("messages")
    ai_messages = [message.content for message in messages if isinstance(message, AIMessage)]


    return ai_messages[-1]