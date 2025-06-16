# API 3D Segmentation

Cette API permet de segmenter des objets 3D avec texture..

## Technologies utilisées

- Python 3.10
- FastAPI pour l’API web
- OpenCV, NumPy, scikit-learn pour le traitement d’image
- Open3D pour le rendu et le traitement 3D
- Deepskin (librairie externe via GitHub)

## Lancer le projet localement

```bash
pip install -r requirements.txt
uvicorn main:app --reload