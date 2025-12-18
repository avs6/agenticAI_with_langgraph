from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
#model = ChatGoogleGenerativeAI(model="gemini-3-pro-preview")

prompt = "what is the captial City of USA?"
result = model.invoke(prompt)
#print(result)
print(result.content)