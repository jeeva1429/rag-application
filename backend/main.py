import os
from fastapi import FastAPI,Form,UploadFile,HTTPException
from fastapi.responses import JSONResponse
from rag_folder.indexer import get_relevant_passage,generate_respose, make_rag_prompt
import shutil
app = FastAPI()

UPLOAD_DIR = "./uploaded_files/"
@app.post("/upload/")
async def upload_file(file: UploadFile):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as filebuffer:
        shutil.copyfileobj(file.file, filebuffer)

    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")

    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "size": file.size,
        "location": str(file_path)
    }        

@app.post("/query/")
async def query_file(query: str = Form(...)):
    print("Received query....:", query)
    get_relevant_passage_result = get_relevant_passage(query, k=3)
    rag_prompt = make_rag_prompt(query, get_relevant_passage_result)
    response_text = generate_respose(rag_prompt)
    return JSONResponse(content={"response": response_text})

@app.get("/")
def root():
    return {"message": "Hello from FastAPI!"}
