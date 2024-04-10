from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from image_text.image2Text import Image2TextModel
import io

app = FastAPI()


@app.post("/image_description")
async def process_image(file:UploadFile):
    model = Image2TextModel()
    request_object_content = await file.read()
    buffer = io.BytesIO(request_object_content)
    result = model.image2TextLocal(buffer)
    return {"result":result}




#uvicorn server:app --host 0.0.0.0 --port 8080
#uvicorn server:app --host 127.0.0.1 --port 8080