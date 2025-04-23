from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
import ollama
import asyncio

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
        model="phi3",
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