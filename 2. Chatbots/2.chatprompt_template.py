from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pprint import pprint
from common import get_llm

model = get_llm()  # uses .env: LLM_PROVIDER / LLM_MODEL

chat_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful {domain} expert"),
    ("human", "Explain in simple terms the concept of {topic}")
])

prompt = chat_template.invoke({
    "domain": "quantum physics",
    "topic": "wormhole"
})

print("---- PROMPT ----")
print(prompt)

response = model.invoke(prompt)

print("\n---- RESPONSE TEXT ----")
print(response.content)

print("\n---- FULL RESPONSE OBJECT ----")
pprint(response.__dict__)
