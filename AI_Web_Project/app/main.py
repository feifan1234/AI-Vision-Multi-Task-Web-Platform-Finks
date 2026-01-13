from fastapi import FastAPI, UploadFile, File, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from PIL import Image
import time
# from app.model import predict_dispatch # Assuming running from parent dir
# Or relative import if running inside app
try:
    from app.model import predict_dispatch
except ImportError:
    from model import predict_dispatch

app = FastAPI(title="AI Image Classification Service")

# 模板目录
templates = Jinja2Templates(directory="app/templates")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )


@app.post("/predict")
async def predict_image(
    file: UploadFile = File(...),
    task: str = Form(...),
    model: str = Form(...)
):
    start_time = time.time()
    try:
        image = Image.open(file.file).convert("RGB")
        results = predict_dispatch(image, task, model)
    except Exception as e:
        return {"error": str(e)}
    end_time = time.time()
    
    if isinstance(results, dict):
        results["time"] = f"{end_time - start_time:.4f}"
        
    return results
