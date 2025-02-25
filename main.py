# from fastapi import FastAPI
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from pydantic import BaseModel
# import torch

# app = FastAPI()

# # Load the model and tokenizer
# model_path = "C:/Users/gowth/coding/college-final-year-project/Hirematrix-chatbot/mistral_finetuned_model"
# tokenizer = AutoTokenizer.from_pretrained(model_path)
# model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32)

# class InputData(BaseModel):
#     text: str

# @app.post("/generate")
# async def generate_text(data: InputData):
#     inputs = tokenizer(data.text, return_tensors="pt")
#     output = model.generate(**inputs, max_length=150)
#     response = tokenizer.decode(output[0], skip_special_tokens=True)
#     return {"response": response}

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="127.0.0.1", port=8000)


# from fastapi import FastAPI
# from pydantic import BaseModel
# from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch

# # Define request model
# class RequestModel(BaseModel):
#     prompt: str
#     max_tokens: int = 128  # Default value

# app = FastAPI()

# MODEL_PATH = "C:/Users/gowth/coding/college-final-year-project/Hirematrix-chatbot/mistral_finetuned_model"

# # Load tokenizer and model on CPU
# tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
# model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="cpu", torch_dtype=torch.float32)

# @app.get("/")
# def home():
#     return {"message": "Mistral 7B API is running!"}

# @app.post("/generate/")
# def generate(request: RequestModel):  # Accept JSON body
#     inputs = tokenizer(request.prompt, return_tensors="pt")

#     with torch.no_grad():  # Disable gradients for inference
#         output = model.generate(**inputs, max_new_tokens=request.max_tokens)

#     response = tokenizer.decode(output[0], skip_special_tokens=True)
#     return {"response": response}

# from fastapi import FastAPI
# import ollama

# app = FastAPI()

# @app.post("/generate")
# def generate(prompt: str):
#     response = ollama.chat(model="hirematrix-mistral", messages=[{"role": "user", "content": prompt}])
#     return {"response": response["message"]["content"]}


from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
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
