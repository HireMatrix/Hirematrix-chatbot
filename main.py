from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
import ollama
import asyncio
import re
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PromptRequest(BaseModel):
    prompt: str

async def generate_stream(prompt: str):
    response = ollama.chat(
        model="hirematrix-mistral",
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )

    for chunk in response:
        yield chunk["message"]["content"]
        await asyncio.sleep(0)

@app.post("/generate")
async def generate(request: PromptRequest):
    return StreamingResponse(generate_stream(request.prompt), media_type="text/plain")

@app.post("/parse-job")
async def parse_job(prompt: PromptRequest):
    final_prompt = f'''
You are a strict JSON generator and not a conversational AI. Extract job data from the input and return exactly one valid JSON array with **one object** only.

‼️ IMPORTANT RULES:
- No explanation, markdown, or extra text.
- Escape all special characters properly.
- No trailing commas.
- Use **double quotes** for all keys and values.
- Arrays: `workMode`, `workType`, `workShift`, `department`
- Use empty string `""` for missing strings, `0` for missing numbers, `[]` for missing arrays.

EXPECTED OUTPUT FORMAT:
[
{{
    "title": "",                    
    "company": "",                  
    "experience": 0,               
    "salary": 0,                   
    "highestEducation": "",        
    "workMode": [],                
    "workType": [],                
    "workShift": [],               
    "department": [],              
    "englishLevel": "",            
    "gender": "",                  
    "location": "",                
    "description": ""              
}}
]

EXTRACTION RULES:
- experience: Extract years from phrases like “3+ years” → 3
- salary: Extract numbers only, e.g. “₹12,00,000” → 1200000
- gender: "Male", "Female", or "Any" (default = "Any")
- englishLevel: "Good English", "Intermediate English", "Advanced English" (default = "Good English")
- Normalize workMode: ["Work from office", "Work from home", "Hybrid"]
- Normalize workType: ["Full time", "Part time", "Internship"]
- Normalize workShift: ["Day shift", "Night shift", "Rotational"]
- description: Use as-is, without paraphrasing

INPUT DATA:
{prompt.prompt}
'''

    response = ollama.chat(
        model="llama3.1",
        messages=[
            { "role": "user", "content": final_prompt }
        ]
    )

    return JSONResponse(content=response["message"]["content"])

@app.post("/embedding")
async def generate_embedding(data: PromptRequest):
    response = ollama.embeddings(
        model='nomic-embed-text',
        prompt=data.prompt
    )
    return JSONResponse(content={"embedding": response["embedding"]})

@app.post("/resumeAnalyzer")
async def resume_analyzer(prompt: PromptRequest):
    final_prompt = f"""
You are a professional resume analyzer.

Analyze the following resume text carefully.

You must strictly return a valid JSON object with the exact following keys and structure (no missing keys, no additional keys):

- summary: (string) A brief 1-2 sentence overview of the candidate's technical strengths.
- skills_detected: (array of strings) List of technical skills explicitly mentioned in the resume.
- experience_level: (string) One of "Entry", "Mid", or "Senior", based on the years of experience and job titles.
- job_titles_detected: (array of strings) List of all job titles found under experience or projects.
- strengths: (array of strings) List specific technical or domain strengths. DO NOT mention formatting strengths.
- weaknesses: (array of strings) List missing technical areas or skill gaps. If none, return an empty array [].
- suggestions: (array of strings) Professional growth suggestions (focus only on technical, certifications, or project improvements).
- grammar_issues: (array of objects) Each object must have:
    - sentence: (string) The sentence with grammar issue.
    - issue: (string) What the grammar mistake is.
- resume_score: (integer) Score the resume out of 100 based on technical skills, experience, and grammar. 
  ( >90 = Excellent, 80-90 = Good, 70-80 = Average, <70 = Needs Improvement)

Example format for grammar_issues:
[
  {{
    "sentence": "Developed scalable web app.",
    "issue": "Missing article 'the'.",
    "updatedSentence": "Developed the scalable web app."
  }},
  {{
    "sentence": "Working on scalable applications.",
    "issue": "Incorrect verb tense.",
    "updatedSentence": "Working on the scalable applications."
  }}
]

Important Rules:
- Only output the JSON. No extra explanations, greetings, or closing remarks.
- Never comment about formatting, font, white space, resume layout, or design.
- Always include all fields, even if some lists are empty (e.g., weaknesses, grammar_issues).
- Ensure the JSON is syntactically valid (no missing commas, brackets, etc).

Resume Text:
{prompt.prompt}
"""
    response = ollama.chat(
        model="llama3.1",
        messages=[
            {"role": "user", "content": final_prompt}
        ],
        options={
            "temperature": 0
        }
    )
    raw_content = response["message"]["content"]

    cleaned_content = re.sub(r"^```(?:json)?\n?", "", raw_content)
    cleaned_content = re.sub(r"\n?```$", "", cleaned_content)

    try: 
        parsed_content = json.loads(cleaned_content)
    except json.JSONDecodeError as e:
        parsed_content = {
            "error": "Invalid JSON response from Ollama. Please try again.",
            "details": str(e),
            "raw_content": cleaned_content
        }

    return JSONResponse(content=parsed_content)
