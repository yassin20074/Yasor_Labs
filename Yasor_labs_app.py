import io
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
import onnxruntime as ort

app = FastAPI(
    title="Yasor Medical-Vision API",
    description="Production-grade API for Chest X-ray diagnosis by Yasor Labs.",
    version="1.0.0"
)

 
ONNX_MODEL_PATH = "chest_xray_resnet50.onnx"
try:
    ort_session = ort.InferenceSession(ONNX_MODEL_PATH, providers=['CPUExecutionProvider'])
except Exception as e:
    raise RuntimeError(f"خطأ في تحميل موديل الـ ONNX: {str(e)}")

CLASSES = ['NORMAL (سليم)', 'PNEUMONIA (التهاب رئوي)'][span_9](start_span)[span_9](end_span)

 
def preprocess_image(image_bytes: bytes) -> np.ndarray:
    try:
        
        raw_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')[span_12](start_span)[span_12](end_span)
         
        resized_img = raw_image.resize((224, 224))
        
        
        img_np = np.array(resized_img).astype(np.float32) / 255.0
        img_np = np.transpose(img_np, (2, 0, 1))
        
         
        mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)[span_16](start_span)[span_16](end_span)
        std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)[span_17](start_span)[span_17](end_span)
        img_np = (img_np - mean) / std
          
        img_np = np.expand_dims(img_np, axis=0).astype(np.float32)
        return img_np
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"فشل معالجة صورة الأشعة: {str(e)}")

 
def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)

@app.post("/predict", summary="Diagnose Chest X-ray Image")
async def predict(file: UploadFile = File(...)):
 
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="الملف المرفوع ليس صورة أشعة صالحة.")
    
     
    image_bytes = await file.read()
    input_tensor = preprocess_image(image_bytes)
    
 
    onnx_inputs = {ort_session.get_inputs()[0].name: input_tensor}
    onnx_outputs = ort_session.run(None, onnx_inputs)
    
  
    logits = onnx_outputs[0]
    probabilities = softmax(logits)[0][span_22](start_span)[span_22](end_span)
    
    pred_idx = int(np.argmax(probabilities))
    confidence = float(probabilities[pred_idx])
    
 
    return {
        "status": "success",
        "company": "Yasor Labs",
        "data": {
            "prediction_index": pred_idx,
            "diagnosis": CLASSES[pred_idx][span_23](start_span)[span_23](end_span),
            "confidence_percentage": round(confidence * 100, 2)[span_24](start_span)[span_24](end_span)
        }
    }

@app.get("/health", summary="API Health Check")
def health_check():
    return {"status": "healthy", "model_loaded": True}