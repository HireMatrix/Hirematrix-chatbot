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

@app.post("/parse-job")
async def parse_job(prompt: PromptRequest):
    final_prompt = f"""
You are a strict JSON generator. Extract job data and return exactly one object inside a JSON array. DO NOT include explanations, markdown, or extra text. should follow this order and in the json format as below.

### OUTPUT FORMAT (ONLY this allowed):
[
{{
    "title": string,
    "experience": number,               // years only (e.g., 2) extract this from the jobtags must be number
    "salary": number,                   // only numeric salary (e.g., 120000) must be number
    "highestEducation": string,        // e.g., "Graduate", "Post Graduate", etc.
    "workMode": [string],              // Only: "Work from office", "Work from home", "Hybrid"
    "workType": [string],              // Only: "Full time", "Part time"
    "workShift": [string],             // Only: "Day shift", "Night shift"
    "department": [string],            // Always an array, can have multiple like ["IT", "Security"]
    "englishLevel": string,            // Only: "Good English", "Intermediate English", "Advanced English"
    "gender": string,                  // Only: "Any", "Male", "Female" these values
    "location": string,                // City, State
    "description": string              // Exact copy from the description
}}
]

### RULES:
- Return ONLY a single array with one object. No markdown, no backticks, no extra text.
- If a field is missing:
  - string → ""
  - number → 0
  - array → []
- Experience: extract numbers from phrases like “3+ years” → 3
- Salary: extract numeric part from strings like “₹12,00,000” → 1200000
- `gender` must be one of: "Male", "Female", "Any" (if not mentioned, default to "Any")
- `englishLevel` must be normalized to:
  - "Good English"
  - "Intermediate English"
  - "Advanced English"
  (if unclear, default to "Good English")
- Normalize work-related fields:
  - Work Mode: "Office", "Work from Office" → "Work from office"; "Remote", "Work from Home" → "Work from home"
  - Work Type: "Full Time" → "Full time"
  - Work Shift: "Day Shift" → "Day shift"
- `department`, `workType`, `workMode`, `workShift` must ALWAYS be arrays, even if only one item.
- `description` must be copied exactly from the source description with no changes or paraphrasing.

### INPUT:
{prompt.prompt}
"""
    response = ollama.chat(
        model="nuextract",
        messages=[
            { "role": "user", "content": final_prompt }
        ]
    )
    
    return JSONResponse(content=response["message"]["content"])

@app.post("/generate")
async def generate(request: PromptRequest):
    return StreamingResponse(generate_stream(request.prompt), media_type="text/plain")
