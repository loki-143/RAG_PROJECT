"""
Simple Gemini Chat - Works with gemini-2.5-flash
"""
import os
from dotenv import load_dotenv
load_dotenv()

import google.generativeai as genai

api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

print("=" * 60)
print("Gemini Chat (gemini-2.5-flash)")
print("Type 'quit' to exit")
print("=" * 60)

model = genai.GenerativeModel("gemini-2.5-flash")
chat = model.start_chat(history=[])

while True:
    try:
        user_input = input("\nYou: ").strip()
        
        if not user_input:
            continue
        if user_input.lower() == 'quit':
            print("Bye!")
            break
            
        response = chat.send_message(user_input)
        print(f"\nBot: {response.text}")
        
    except KeyboardInterrupt:
        print("\nBye!")
        break
    except Exception as e:
        print(f"\nError: {e}")
