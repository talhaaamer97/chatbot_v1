# 1. setup pydantic model
from typing import List
from pydantic import BaseModel
# for communitcation we need a standard
# kis tarhan ki info aey gi, kis form mein
# isi lieye class banaein gay, req state inherit karay gi
# basemodel pydantic say. us k upar communication apni build kariein gay
# front aur backend k bech communication ki
# in easier terms, user is type ki info de ga tab hi processing shuru karein gay

class RequestState(BaseModel):
    model_name: str
    model_provider: str
    system_prompt: str
    messages: List[str]
    allow_search: bool

# 2. setup AI agent from front end request
# the access to models we have via Groq api
ALLOWED_MODEL_NAMES= ["llama3-70b-8192", "mixtral-8x7b-32768","llama-3.3-70b-versatile","gpt-4o-mini"]

from fastapi import FastAPI
from ai_agent import get_response_from_ai_agent

# end point processing
app=FastAPI(title='Langgrah AI Agent')

# post request aie front end say us ko is "chat" end point par recieve karna hay
@app.post('/chat')
# how to process this
def chat_endpoint(request: RequestState): #request ki data type class reqstate ho gi
    """
    Api endpoint to interact with chatbot using LangGraph and search tools
    it dynamically selects the model specified in the request 
    """
    if request.model_name not in ALLOWED_MODEL_NAMES:
        return {"error" : "invalid model name, select available model name"}
    
    model_name = request.model_name
    query = request.messages
    allow_search = request.allow_search
    system_prompt = request.system_prompt
    provider = request.model_provider

    # create AI agent and get response from it
    response = get_response_from_ai_agent(
        model_name=model_name,
        query=query,
        allow_search=allow_search,
        system_prompt=system_prompt,
        provider=provider
        )
    return response
# endpoint banana hay jis par humara front end msg bhej sakay ga

#step 3 run app and explore swagger ui docs
if __name__=="__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=9999)

