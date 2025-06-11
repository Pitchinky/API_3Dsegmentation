from fastapi import FastAPI, UploadFile, File, Form 
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import uuid
from segmentation import process_obj_with_texture
from fastapi.staticfiles import StaticFiles

app = FastAPI()
app.mount("/models", StaticFiles(directory="models"), name="models")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Ici j'ajoute le .mtl pour chaque modèle (à adapter selon tes fichiers)
PREDEFINED_MODELS = {
    "model1": ("models/model1.obj", "models/model1.mtl", "models/model1.png"),
    "model2": ("models/model2.obj", "models/model2.mtl", "models/model2.png"),
    "model3": ("models/model3.obj", "models/model3.mtl", "models/model3.png"),
    "model4": ("models/model4.obj", "models/model4.mtl", "models/model4.png"),
}

@app.get("/models")
def get_models():
    return list(PREDEFINED_MODELS.keys())

@app.post("/segment")
async def segment(
    method: str = Form(None),
    model_name: str = Form(None),
    obj_file: UploadFile = File(None),
    texture_file: UploadFile = File(None)
):
    if model_name:
        obj_path, _, texture_path = PREDEFINED_MODELS.get(model_name, (None, None, None))
        if not obj_path:
            return JSONResponse({"error": "Modèle introuvable"}, status_code=404)
    else:
        if not obj_file or not texture_file:
            return JSONResponse({"error": "Fichiers manquants"}, status_code=400)

        obj_path = f"uploads/{uuid.uuid4().hex}.obj"
        texture_path = f"uploads/{uuid.uuid4().hex}.png"
        with open(obj_path, "wb") as f:
            shutil.copyfileobj(obj_file.file, f)
        with open(texture_path, "wb") as f:
            shutil.copyfileobj(texture_file.file, f)

    output_dir = f"outputs/{uuid.uuid4().hex}"
    os.makedirs(output_dir, exist_ok=True)

    obj_out, img_out, mtl_out = process_obj_with_texture(obj_path, texture_path, output_dir, method)

    return {
        "obj_url": f"/download/{os.path.basename(output_dir)}/output.obj",
        "mtl_url": f"/download/{os.path.basename(output_dir)}/material.mtl",
        "texture_url": f"/download/{os.path.basename(output_dir)}/segmented_texture.png"
    }

@app.post("/preview")
async def preview(
    model_name: str = Form(None),
    obj_file: UploadFile = File(None),
    mtl_file: UploadFile = File(None)
):
    """
    Endpoint simple pour prévisualiser un modèle 3D sans segmentation.
    Soit on choisit un modèle prédéfini,
    soit on upload un obj + mtl.
    """
    if model_name:
        obj_path, mtl_path, _ = PREDEFINED_MODELS.get(model_name, (None, None, None))
        if not obj_path or not mtl_path:
            return JSONResponse({"error": "Modèle introuvable"}, status_code=404)
    else:
        if not obj_file or not mtl_file:
            return JSONResponse({"error": "Fichiers manquants"}, status_code=400)
        obj_path = f"uploads/{uuid.uuid4().hex}.obj"
        mtl_path = f"uploads/{uuid.uuid4().hex}.mtl"
        with open(obj_path, "wb") as f:
            shutil.copyfileobj(obj_file.file, f)
        with open(mtl_path, "wb") as f:
            shutil.copyfileobj(mtl_file.file, f)

    return {
        "obj_url": f"/models/{os.path.basename(obj_path)}",
        "mtl_url": f"/models/{os.path.basename(mtl_path)}"
    }

@app.get("/download/{folder}/{filename}")
def download_file(folder: str, filename: str):
    file_path = f"outputs/{folder}/{filename}"
    return FileResponse(file_path)

@app.get("/model/{model_name}")
def get_model(model_name: str):
    return {
        "obj_url": f"/models/{model_name}.obj",
        "mtl_url": f"/models/{model_name}.mtl"
    }
