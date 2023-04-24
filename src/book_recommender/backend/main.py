import json

import lc_utils
from fastapi import FastAPI
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.schema import SystemMessage, messages_from_dict, messages_to_dict
from loguru import logger

app = FastAPI()


@app.get("/")
async def health_check():
    return {"status": "OK!"}


@app.post("/chat/{open_ai_model}")
async def chat(
    open_ai_model: lc_utils.ModelName, messages: list[lc_utils.Message], query: str
):
    # Step 1: Parse chat history
    memory = ConversationBufferMemory()
    memory.chat_memory.messages = messages_from_dict(
        [message.dict() for message in messages]
    )

    # Step 2: Add the system prompt
    system_prompt = SystemMessage(
        content="You are a helpful virtual bibliophile assisstant. You help to recommend books to users based on their queries. If the question is not related to books you politely decline to answer."
    )
    memory.chat_memory.messages = [system_prompt] + memory.chat_memory.messages

    # Step 3: Create the response prompt
    query = f"""Respond only in the following json format: {{"response": str, "list_of_books": [str]}} and do not add any additional text outside this format.\n\nQuery: {query}"""
    logger.info(query)

    # Step 4: Initialize the model chain
    model_name = lc_utils.open_ai_model(open_ai_model)
    llm = ChatOpenAI(model_name=model_name, temperature=0.5, request_timeout=30)
    conversation = ConversationChain(llm=llm, memory=memory)

    # Step 5: Generate a response
    response = conversation.predict(input=query)
    logger.info(response)

    # Step 6: Parse the response
    memory.chat_memory.messages[-1].content = json.loads(
        memory.chat_memory.messages[-1].content
    )

    return messages_to_dict(memory.chat_memory.messages)[-1]
