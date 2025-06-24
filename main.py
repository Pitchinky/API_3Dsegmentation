from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import shutil
import os
import uuid
from segmentation import process_obj_with_texture

app = FastAPI()

# Monter les dossiers statiques
app.mount("/models", StaticFiles(directory="models"), name="models")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

PREDEFINED_MODELS = {
    "model1": ("models/model1.obj", "models/model1.mtl", "models/model1.png"),
    "model2": ("models/model2.obj", "models/model2.mtl", "models/model2.png"),
    "model3": ("models/model3.obj", "models/model3.mtl", "models/model3.png"),
    "model4": ("models/model4.obj", "models/model4.mtl", "models/model4.png"),
    "model5": ("models/model5.obj", "models/model5.mtl", "models/model5.png"),
}

@app.get("/models")
def get_models():
    return list(PREDEFINED_MODELS.keys())

@app.post("/segment")
async def segment(
    method: str = Form(None),
    model_name: str = Form(None),
    obj_file: UploadFile = File(None),
    texture_file: UploadFile = File(None),
    mtl_file: UploadFile = File(None)
):
    if model_name:
        obj_path, mtl_path, texture_path = PREDEFINED_MODELS.get(model_name, (None, None, None))
        if not obj_path:
            return JSONResponse({"error": "Modèle introuvable"}, status_code=404)
    else:
        if not obj_file or not texture_file:
            return JSONResponse({"error": "Fichiers manquants"}, status_code=400)

        # Créer un dossier unique pour les fichiers uploadés
        folder_id = uuid.uuid4().hex
        upload_dir = os.path.join("uploads", folder_id)
        os.makedirs(upload_dir, exist_ok=True)

        # Sauvegarde des fichiers avec leur nom d'origine
        obj_path = os.path.join(upload_dir, obj_file.filename)
        texture_path = os.path.join(upload_dir, texture_file.filename)
        mtl_path = os.path.join(upload_dir, mtl_file.filename) if mtl_file else None

        with open(obj_path, "wb") as f:
            shutil.copyfileobj(obj_file.file, f)
        with open(texture_path, "wb") as f:
            shutil.copyfileobj(texture_file.file, f)
        if mtl_file:
            with open(mtl_path, "wb") as f:
                shutil.copyfileobj(mtl_file.file, f)

    # Créer le dossier de sortie
    output_dir = f"outputs/{uuid.uuid4().hex}"
    os.makedirs(output_dir, exist_ok=True)

    # Appeler la fonction de traitement avec le chemin .mtl si disponible
    obj_out, img_out, mtl_out, pwat = process_obj_with_texture(
        obj_path, texture_path, output_dir, method, mtl_path
    )

    return {
        "obj_url": f"/download/{os.path.basename(output_dir)}/output.obj",
        "mtl_url": f"/download/{os.path.basename(output_dir)}/material.mtl",
        "texture_url": f"/download/{os.path.basename(output_dir)}/segmented_texture.png",
        "pwat": pwat
    }


@app.post("/preview")
async def preview(
    model_name: str = Form(None),
    obj_file: UploadFile = File(None),
    texture_file: UploadFile = File(None),
    mtl_file: UploadFile = File(None)
):
    if model_name:
        obj_path, mtl_path, texture_path = PREDEFINED_MODELS.get(model_name, (None, None, None))
        if not obj_path or not mtl_path:
            return JSONResponse({"error": "Modèle introuvable"}, status_code=404)
        return {
            "obj_url": f"/{obj_path}",
            "mtl_url": f"/{mtl_path}",
            "texture_url": f"/{texture_path}"
        }

    # Sinon, l'utilisateur a uploadé ses propres fichiers
    if not obj_file or not texture_file:
        return JSONResponse({"error": "Fichiers manquants"}, status_code=400)

    # Créer un sous-dossier unique dans uploads/
    folder_id = uuid.uuid4().hex
    upload_dir = os.path.join("uploads", folder_id)
    os.makedirs(upload_dir, exist_ok=True)

    # Sauvegarder les fichiers avec leur nom d'origine
    obj_path = os.path.join(upload_dir, obj_file.filename)
    texture_path = os.path.join(upload_dir, texture_file.filename)
    mtl_path = os.path.join(upload_dir, mtl_file.filename) if mtl_file else None

    with open(obj_path, "wb") as f:
        shutil.copyfileobj(obj_file.file, f)
    with open(texture_path, "wb") as f:
        shutil.copyfileobj(texture_file.file, f)
    if mtl_file:
        with open(mtl_path, "wb") as f:
            shutil.copyfileobj(mtl_file.file, f)

    return {
        "obj_url": f"/uploads/{folder_id}/{obj_file.filename}",
        "texture_url": f"/uploads/{folder_id}/{texture_file.filename}",
        "mtl_url": f"/uploads/{folder_id}/{mtl_file.filename}" if mtl_file else None
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
