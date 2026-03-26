from dotenv import load_dotenv
import os
load_dotenv(override=True)

openai_api_key = os.getenv('API_TOKEN')
print(openai_api_key)