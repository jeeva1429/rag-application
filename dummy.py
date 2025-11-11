# Generate answer using Gemini LLM
from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()
OPEN_AI_API_KEY = os.getenv("OPENAI_API_KEY")
def generate_respose(prompt):
    # client = genai.Client()
    client = OpenAI(api_key=OPEN_AI_API_KEY)
    try:
        response = client.responses.create(model="gpt-5-nano", input=prompt,)
        return response.output_text
        # response  = client.models.generate_content(model="gemini-2.0-flash",contents=prompt)
        # return response.text
    except Exception as e:
        return f"Error generating response: {str(e)}"
    


res = generate_respose("Hello, what is the capital of USA?")
print(res)