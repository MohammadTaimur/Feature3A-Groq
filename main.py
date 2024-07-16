"""pip install groq torch torchvision torchaudio spacy
pip install python-dotenv
pip install -U git+https://github.com/PrithivirajDamodaran/Gramformer.git
python -m spacy download en_core_web_sm
pip install fastapi uvicorn"""
from groq import Groq
from dotenv import load_dotenv
import os
import torch
from gramformer import Gramformer
from fastapi import FastAPI, HTTPException, Form

load_dotenv()

API_KEY = os.getenv("GROQ_PRACTICE")

client = Groq(api_key=API_KEY)

app = FastAPI()


@app.post("/generate-answers/")
async def generate_answers(
        question: str = Form(None)
):
    if question is None:
        raise HTTPException(status_code=400, detail="Missing question")
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": """ You are an expert in teaching English to Foreign Beginners. Your task is to give any 5 generic answers to any type of question asked.

                     Instructions:
                     1. Input: 
                     - A short question in the form of a string. For example: "What is your favorite food?"

                     2. Processing Instructions:
                     - Generate 5 different answers.
                     - Each answer should be line by line separately.
                     - Answer every part of the question (if more than 2).
                     - Frame each answer as a personal preference. For example: "I like pizza", "I like chicken and rice", etc.

                     3. Output Format: 
                     - Return the answers in the following format: "I like pizza", "I like chicken and rice", ... (up to 5 answers).
                     - Provide only the requested answers, with no extra text, or introductory statements.
                     """
                },
                {
                    "role": "user",
                    "content": question,
                }
            ],
            model="llama3-70b-8192",
        )
        output = chat_completion.choices[0].message.content
        # Split the input string into lines
        lines = output.strip().split('\n')

        # Assign each line to a standalone variable
        option1 = lines[0]
        option2 = lines[1]
        option3 = lines[2]
        option4 = lines[3]
        option5 = lines[4]

        # Print the variables to verify
        print(option1)
        print(option2)
        print(option3)
        print(option4)
        print(option5)

        result = {
            "option1" : option1,
            "option2" : option2,
            "option3" : option3,
            "option4" : option4,
            "option5" : option5,
        }

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")
# lines = output.splitlines()

# print(lines[0])
def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(1212)

@app.post("/check-grammar/")
async def check_grammar(
        question: str = Form(None)
):
    if question is None:
        raise HTTPException(status_code=400, detail="Missing question")
    try:
        gf = Gramformer(models=1, use_gpu=False)  # 1=corrector, 2=detector

        corrections = gf.correct(question)
        correction = list(corrections)[0]

        if correction == question:
            return {
                "grammar": ""
            }

        else:
            result = {
                "grammar": correction
            }
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error in processing")