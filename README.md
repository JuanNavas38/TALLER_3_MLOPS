# Taller 2: Palmer Penguins - MLOps

**MLOps - Pontificia Universidad Javeriana**

**Clasificación de especies de pingüinos usando Docker Compose y `uv`.**

## Autores
*   **Juan Navas**
*   **Camila Cuellar**
*   **Jhonathan Murcia**

---

## Descripción General
Este proyecto evoluciona el taller anterior hacia una arquitectura de contenedores orquestada con Docker Compose. Se ha integrado el gestor de paquetes `uv` para una gestión de dependencias moderna y eficiente, y el sistema permite tanto la experimentación en Jupyter como el despliegue de una API de producción compartiendo la misma lógica y datos.

## Estructura del Proyecto

```text
TALLER_2_MLOPS/
├── src/                                # Código fuente
│   ├── config.py                       # Configuración (Data Path, Model Path)
│   ├── preprocessing.py                # Limpieza y encoding
│   ├── train.py                        # Entrenamiento
│   └── api.py                          # FastAPI
├── data/                               # Dataset local (penguins.csv)
├── notebooks/                          # Jupyter Notebooks
│   └── train_model.ipynb
├── models/                             # Artefactos (PKL, Metrics)
├── docker-compose.yml                  # Orquestación de servicios
├── Dockerfile                          # Imagen de la API
├── Dockerfile.jupyter                  # Imagen de JupyterLab
├── pyproject.toml                      # Gestión de dependencias con uv
└── README.md
```

## Ejecución con Docker Compose (Recomendado)

### 1. Levantar los servicios
```powershell
docker-compose up --build -d
```
Esto iniciará:
*   **JupyterLab**: [http://localhost:8888](http://localhost:8888) (Acceso sin token)
*   **API**: [http://localhost:8989](http://localhost:8989) (Docs: [/docs](http://localhost:8989/docs))

### 2. Entrenamiento de Modelos
El entrenamiento se puede disparar desde cualquier entorno:
*   **Desde la terminal**: `docker-compose exec jupyter uv run python -m src.train`
*   **Desde Jupyter**: Ejecutando las celdas de `notebooks/train_model.ipynb`.

### 3. Predicciones vía API
Puedes probar la clasificación usando el Swagger UI o `curl`:
```powershell
curl -X POST "http://localhost:8989/predict/rf" `
     -H "Content-Type: application/json" `
     -d '{
           "island": "Biscoe",
           "bill_length_mm": 45.0,
           "bill_depth_mm": 15.0,
           "flipper_length_mm": 210.0,
           "body_mass_g": 4000.0,
           "sex": "male"
         }'
```

---

## Desarrollo Local (Opcional)
Si tienes `uv` instalado localmente:
```bash
uv sync
uv run python -m src.train
uv run python -m src.api
```
