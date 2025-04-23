import os
import requests
from flask import Flask, request, jsonify
import pandas as pd
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# I am sure you'll have an API key laying around. Otherwise trust me, it works😉
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)
llm = ChatOpenAI(model_name="gpt-4", temperature=0.7, openai_api_key=OPENAI_API_KEY)

# Define structured output model
class PersonaResponse(BaseModel):
    name: str = Field(description="The persona's name")
    age: str = Field(description="The persona's age")
    selected_gender: str = Field(description="The persona's gender")
    location: str = Field(description="The persona's exact location")
    persona_information: str = Field(description="The information about the generated persona")

# Prompt template
PROMPT_PERSONA_CREATION = """
Create a detailed fictional persona based on the following:
- Age Range: {age_range}
- Region: {region}
- Gender: {gender}

The persona should include personality traits, lifestyle, interests, and tone consistent with those traits.
Please return the output in the following JSON-like format:
{format_instructions}
"""

# Allows to create n personas based on the given input. 
# This is a very rudementary setup but you can see how this can be tailored for a specific purpose.
# Results are stored in a csv. You would obviously not do this, trust me I could also do it with a DB.
@app.route("/create_persona", methods=["POST"])
def create_persona():
    data = request.get_json()

    age_range = data.get("age_range")
    region = data.get("region")
    gender = data.get("gender")
    n = data.get("n", 1)

    if not all([age_range, region, gender]):
        return jsonify({"error": "Missing age, location, or gender"}), 400

    if not isinstance(n, int) or n < 1:
        return jsonify({"error": "n must be a positive integer"}), 400

    personas = []

    parser = PydanticOutputParser(pydantic_object=PersonaResponse)
    message = HumanMessagePromptTemplate.from_template(PROMPT_PERSONA_CREATION)
    chat_prompt = ChatPromptTemplate.from_messages([message])

    personas = []
    # For now if the OPENAI API does not return a valid persona ignore it and just keep going.
    for _ in range(n):
        try:
            prompt = chat_prompt.format_prompt(
                age_range=age_range,
                region=region,
                gender=gender,
                format_instructions=parser.get_format_instructions()
            )
            output = llm.invoke(prompt.to_messages())
            persona = parser.parse(output.content).model_dump()
            personas.append(persona)
        except Exception as e:
            return jsonify({"message": 'error in creating persona', "error": str(e)}), 500

    # If the csv db does not exist yet make sure to create it
    if os.path.exists('personas.csv'):
        df = pd.read_csv('personas.csv')
    else:
        df = pd.DataFrame(columns=["age", "location", "gender", "persona_information", "age_range", 'region'])

    # Append new personas
    new_df = pd.DataFrame(personas)
    new_df['age_range'] = age_range
    new_df['region'] = region
    new_df['gender'] = gender
    df = pd.concat([df, new_df], ignore_index=True)

    # Save back to CSV
    df.to_csv('personas.csv', index=False)

    return jsonify({
        "message": f"{len(personas)} persona(s) created."
    })
 

# This API is used to actually get the personas. 
# If there are less personas meeting the criterea than required it will create more.
# If more exist it samples.
# Again this is a POC so actual implementation would be more refined. Use some metadata to track counts etc.
# Maybe also do a couple of loops just in case GPT-4 decides to step out of line and return invalid structures.
@app.route("/retrieve_personas", methods=["POST"])
def retrieve_personas():
    data = request.get_json()
    age_range = data.get("age_range")
    region = data.get("region")
    gender = data.get("gender")
    n = int(data.get("n", 1))

    if not all([age_range, region, gender]):
        return jsonify({"error": "Missing age, location, or gender"}), 400

    if os.path.exists('personas.csv'):
        df = pd.read_csv('personas.csv')
        # Filter matching rows
        matching = df[
            (df["age_range"] == age_range) &
            (df["region"] == region) &
            (df["gender"] == gender)
        ]
        n_matching = len(matching)
    else:
        n_matching = 0
    
    print(os.path.exists('personas.csv'))
    print(n_matching)

    if n_matching >= n:
        return jsonify({"personas": matching.sample(n).to_dict(orient="records")})

    # If not enough, request more from the API
    to_generate = n - n_matching
    try:
        response = requests.post(
            "http://localhost:5001/create_persona",
            json={
                "age_range": age_range,
                "region": region,
                "gender": gender,
                "n": to_generate
            }
        )
        if response.status_code != 200:
            return jsonify({"error": "Failed to create new personas", "details": response.json()}), 500
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Request failed: {e}"}), 500
    
    # The csv always exists here.
    df = pd.read_csv('personas.csv')
    matching = df[
        (df["age_range"] == str(age_range)) &
        (df["region"] == region) &
        (df["gender"] == gender)
    ]

    return jsonify({"personas": matching.sample(n).to_dict(orient="records")})


@app.route("/", methods=["GET"])
def home():
    return "LangChain GPT persona wrapper can be reached"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
