import os

import uvicorn

from nlp_ml_lab.serving import app as serving_module
from nlp_ml_lab.serving.app import app
from nlp_ml_lab.serving.loader import load_predictor_from_environment


def main() -> None:
    if os.getenv("NLP_ML_MODEL_PATH", "").strip():
        serving_module.service.predictor = load_predictor_from_environment()

    uvicorn.run(
        app,
        host=os.getenv("NLP_ML_HOST", "0.0.0.0"),
        port=int(os.getenv("NLP_ML_PORT", "8000")),
    )


if __name__ == "__main__":
    main()
