from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from search import search_result, user_query
import os

load_dotenv()
context = "\n\n".join([doc.page_content for doc,score in search_result])


prompt_template = ChatPromptTemplate.from_template(
    """You are a helpful assistant. Answer the question based ONLY on the following context. 
If the context doesn't contain the answer, say "I couldn't find that in the document."

Context:
{context}

Question:
{user_query}

Answer:"""
)

final_prompt = prompt_template.format(context=context, user_query=user_query)

os.environ["GROQ_API_KEY"] = os.getenv('GROQ_API_KEY')



llm = ChatGroq(
    model="qwen/qwen3.6-27b",
    temperature=0,
    max_tokens=400
)

print("Generating answer...\n")
response = llm.invoke(final_prompt)

print("=" * 50)
print("RESPONSE:")
print(response)
print("=" * 50)
