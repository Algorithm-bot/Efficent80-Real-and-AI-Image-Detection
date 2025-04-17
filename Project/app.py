from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
import numpy as np
import io
from PIL import Image

# Load the trained model
model = load_model("ai_image_detector.keras")  # Ensure the correct file path

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup templates
templates = Jinja2Templates(directory="templates")

# Function to preprocess image
def preprocess_image(img):
    img = img.resize((224, 224))  # Resize to match model input
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = preprocess_input(img_array)  # EfficientNet-specific preprocessing
    return img_array

@app.get("/", response_class=HTMLResponse)
async def read_root(request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Read image
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")  # Ensure correct format
        
        # Preprocess image
        img_array = preprocess_image(img)

        # Make prediction
        prediction = model.predict(img_array)
        raw_confidence = float(prediction[0][0])

        # Debugging: Print raw model output
        print(f"Raw prediction: {raw_confidence}")

        # Set classification threshold
        threshold = 0.5  # Adjust if needed

        # Compute confidence directly from probability
        confidence = raw_confidence if raw_confidence > threshold else 1 - raw_confidence

        # Assign label based on threshold
        result = "AI-Generated" if raw_confidence > threshold else "Real Image"

        return {"prediction": result, "confidence": confidence}

    except Exception as e:
        return {"error": str(e)}

# Run server using:
# uvicorn app:app --host 0.0.0.0 --port 8000
