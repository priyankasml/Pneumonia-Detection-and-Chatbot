import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

API_KEY = os.getenv("GROQ_API_KEY")

if not API_KEY:
    raise ValueError("GROQ_API_KEY not found.")

API_KEY = API_KEY.encode("ascii", "ignore").decode().strip()

client = Groq(api_key=API_KEY)


def chatbot_response(user_message, last_prediction=None):

    if not user_message:
        return "Please type your question."

    if last_prediction is None:
        return "Please upload an X-ray first so I can explain your diagnosis."

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": """
You are an experienced radiologist.
Speak professionally and calmly.
Do NOT prescribe medicines.
Encourage doctor consultation.
"""
                },
                {
                    "role": "user",
                    "content": f"""
Diagnosis: {last_prediction['prediction']}
Confidence: {last_prediction['confidence']}%
Stage: {last_prediction['stage']}
Comment: {last_prediction['comment']}

Question: {user_message}
"""
                }
            ],
            temperature=0.4,
            max_tokens=400
        )

        return response.choices[0].message.content.strip()

    except Exception:
        return "⚠️ AI service temporarily unavailable."