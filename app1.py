# import libraries
from fastapi import FastAPI, Form, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import numpy as np
import pickle
import uvicorn

# Create the object
app = FastAPI(debug=True)
templates = Jinja2Templates(directory="C:/Users/shopinverse/Documents/FastAPI Project/Heart Disease/templates")

# Configure the StaticFiles to serve CSS and other static files
app.mount("/static", StaticFiles(directory="C:/Users/shopinverse/Documents/FastAPI Project/Heart Disease/static"), name="static")

# Load the model
file_path = 'C:/Users/Shopinverse/Documents/FastAPI Project/Heart Disease/XGBclassifier.pkl'
with open(file_path, 'rb') as file:
    clf = pickle.load(file)

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("prediction.html", {"request": request, "prediction": ""})

@app.post('/predict')
def predict(
    request: Request,
    currentSmoker: float = Form(...),
    BPMeds: float = Form(...),
    prevalentStroke: float = Form(...),
    prevalentHyp: float = Form(...),
    diabetes: float = Form(...),
    Gender: float = Form(...),
    Scaled_age: float = Form(...),
    Scaled_education: float = Form(...),
    Scaled_CigsPerDay: float = Form(...),
    Scaled_totChol: float = Form(...),
    Scaled_sysBP: float = Form(...),
    Scaled_diaBP: float = Form(...),
    Scaled_BMI: float = Form(...),
    Scaled_heartRate: float = Form(...),
    Scaled_glucose: float = Form(...),
):
    # Extract data from the request
    input_data = np.array([[
        currentSmoker, BPMeds, prevalentStroke, prevalentHyp, diabetes, Gender,
        Scaled_age, Scaled_education, Scaled_CigsPerDay, Scaled_totChol, Scaled_sysBP,
        Scaled_diaBP, Scaled_BMI, Scaled_heartRate, Scaled_glucose
    ]])
   
    prediction = clf.predict(input_data)

    if prediction[0] > 0.5:
        result = "This Patient has Coronary Heart Disease"
    else:
        result = "This Patient does not have Coronary Heart Disease"

    return templates.TemplateResponse("prediction.html", {"request": request, "prediction": result})

# Run the API with uvicorn
# Will run on http://127.0.0.1:8000

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
