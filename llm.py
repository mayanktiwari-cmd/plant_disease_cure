import os
from groq import Groq

# Load from a .env file when present (development only)
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    # python-dotenv is optional; environment variables are preferred in production
    pass

# Read the API key from environment variable GROQ_API_KEY
API_KEY = os.environ.get("GROQ_API_KEY")
if not API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY not set. Set it in your environment or create a .env file with GROQ_API_KEY=..."
    )

client = Groq(api_key=API_KEY)


#Treatment Generator


def get_cure(plant, disease):

   
    if disease.lower() == "healthy":
        return """
 Plant Health Status: Healthy

Recommendations:
• Continue regular watering
• Ensure proper sunlight exposure
• Monitor leaves weekly
• Maintain soil health
• Apply balanced fertilizer when needed
"""

    prompt = f"""
A farmer has a {plant} plant affected by {disease}.

Provide:

1. Cause of the disease (1-2 sentences)
2. Immediate treatment steps (3-5 bullet points)
3. Prevention methods (3-5 bullet points)
4. Whether organic or chemical treatment is recommended

Keep the explanation practical and easy for farmers.

Maximum 150 words.
"""

    try:

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3,
            max_tokens=300
        )

        return response.choices[0].message.content

    except Exception as e:

        return f"""
 Unable to generate AI treatment advice.

Error:
{str(e)}

Basic Recommendations:

• Remove infected leaves immediately
• Avoid overwatering
• Improve air circulation
• Use suitable fungicide/pesticide if required
• Monitor the plant for 7–14 days
"""
