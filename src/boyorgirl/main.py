import pandas as pd
import numpy as np
import os
import re
from typing import List
from tensorflow.keras.models import load_model

import uvicorn
from fastapi import FastAPI
from fastapi.logger import logger

from preprocess import preprocess

# Load the model
model_path = os.path.join(os.path.dirname(__file__), "models/boyorgirl.h5")
pred_model = load_model(model_path)

# Instantiate the app
app = FastAPI()

# Predict endpoint
@app.post("/predict")
def predict(names: List[str]):
    logger.info(names)

    # Split on all non-alphabet characters
    split_names = [re.findall(r"\w+", name) for name in names]
    names = [item for sublist in split_names for item in sublist]

    # Convert to dataframe
    pred_df = pd.DataFrame({"name": names})

    # Preprocess
    pred_df = preprocess(pred_df, train=False)

    # Predictions
    result = pred_model.predict(np.asarray(pred_df["name"].values.tolist())).squeeze(
        axis=1
    )
    pred_df["Boy or Girl?"] = ["Boy" if logit > 0.5 else "Girl" for logit in result]
    pred_df["Probability"] = [logit if logit > 0.5 else 1.0 - logit for logit in result]

    # Format the output
    pred_df["name"] = names
    pred_df.rename(columns={"name": "Name"}, inplace=True)
    pred_df["Probability"] = pred_df["Probability"].round(2)
    pred_df.drop_duplicates(inplace=True)

    return {"response": pred_df.to_json(orient="records")}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
