from fastapi import FastAPI, File, UploadFile
from starlette.responses import JSONResponse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import io
import uvicorn
import os

app = FastAPI()

MODEL_PATH = 'flower_cnn_model.h5'
DATA_DIR = 'flower_images'
model = load_model(MODEL_PATH)
classes = sorted(os.listdir(DATA_DIR))
idx_to_class = {i: cls for i, cls in enumerate(classes)}
IMAGE_SIZE = (128, 128)

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    data = await file.read()
    image = Image.open(io.BytesIO(data)).convert('RGB')
    image = image.resize(IMAGE_SIZE)
    image_array = img_to_array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    prediction = model.predict(image_array)
    class_idx = np.argmax(prediction)
    confidence = float(prediction[0][class_idx])
    label = idx_to_class[class_idx]
    return JSONResponse(content={"label": label, "confidence": confidence})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
