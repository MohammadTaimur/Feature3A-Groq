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
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": """ You are an expert in teaching English to Foreign Beginners. Your task is to give any 5 generic answers to anytype of question asked.
    
                 Instructions:
                 1. Input: 
                 - A short question in the form of a string. For example: "What is your favorite food?"
    
                 2. Processing Instructions:
                 - Generate 5 different answers.
                 - Each answer should be line by line separately.
                 - Answer every part of the question (if more then 2).
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

    result = {
        "generated_answers": output
    }

    return result
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
    gf = Gramformer(models=1, use_gpu=False)  # 1=corrector, 2=detector

    result = {
        "grammar": gf.correct(question)
    }
    return result