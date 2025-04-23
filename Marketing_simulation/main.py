import os
import requests
from flask import Flask, request, jsonify
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# I am sure you'll have an API key laying around. Otherwise trust me, it works😉
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)
llm = ChatOpenAI(model_name="gpt-4", temperature=0.7, openai_api_key=OPENAI_API_KEY)

class PersonaResponse(BaseModel):
    response: str = Field(description="The persona's response to the marketing text")
    sentiment: str = Field(description="The sentiment of the persona's response (positive/negative/neutral)")
    explanation: str = Field(description="Explanation for the sentiment")

# Prompt template
PROMPT_PERSONA_SIMULATION = """
You are the following persona:
- Age: {age}
- Location: {location}
- Gender: {gender}
- Description: {persona_information}

Read this marketing message:
"{marketing_text}"

Based on your persona's traits and interests, how would you realistically react to this message? 
Please also provide a sentiment score (positive, negative, neutral) for your reaction, and explain why you feel that way in your response.
Please return the output in the following JSON-like format:
{format_instructions}
"""

@app.route("/simulate_marketing", methods=["POST"])
def simulate_persona():
    data = request.get_json()
    age_range = data.get("age_range")
    region = data.get("region")
    gender = data.get("gender")
    n = data.get("n", 1)
    marketing_text = data.get("marketing_text")

    if not all([age_range, region, gender, marketing_text]):
        return jsonify({"error": "Missing age_range, region, gender, or marketing_text"}), 400

    # Request a personas from the other app
    try:
        response = requests.post("http://app1:5001/retrieve_personas", json={
        "age_range": age_range,
        "region": region,
        "gender": gender,
        "n": n
        })
        response.raise_for_status()
        persona_data = response.json()
        personas = persona_data["personas"]
        if response.status_code != 200:
            return jsonify({"error": "Failed to retrieve personas", "details": response.json()}), 500
    except Exception as e:
        return jsonify({"error": f"Failed to get persona from persona creator: {e}"}), 500

    parser = PydanticOutputParser(pydantic_object=PersonaResponse)
    message = HumanMessagePromptTemplate.from_template(PROMPT_PERSONA_SIMULATION)
    chat_prompt = ChatPromptTemplate.from_messages([message])

    marketing_results = []
    # For now if the OPENAI API does not return a valid persona ignore it and just keep going.
    # For each of the personas see how they would respond to the marketing message.
    for persona in personas:
        try:
            prompt = chat_prompt.format_prompt(
                age=persona['age'],
                location=persona['location'],
                gender=persona['selected_gender'],
                persona_information=persona['persona_information'],
                marketing_text=marketing_text,
                format_instructions=parser.get_format_instructions()
            )
            output = llm.invoke(prompt.to_messages())
            parsed = parser.parse(output.content).model_dump()
            marketing_results.append(persona | parsed)
        except Exception as e:
            return jsonify({"message": 'error in marketing sim', "error": str(e)}), 500
    
    return jsonify({
        "message": f"{len(marketing_results)} persona(s) had a reaction to your marketing message.",
        "marketing_results": marketing_results
    })

@app.route("/", methods=["GET"])
def home():
    return "LangChain GPT simulation wrapper can be reached"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002)
