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
    # Step 1: Input is a list of names
    logger.info(names)

    # Step 2: Split on all non-alphabet characters
    split_names = [re.findall(r"\w+", name) for name in names]
    names = [item for sublist in split_names for item in sublist]

    # Step 3: Keep only first 10 names
    names = names[:10]

    # Convert to dataframe
    pred_df = pd.DataFrame({"name": names})

    # Step 4: Preprocess the names
    pred_df = preprocess(pred_df)

    # Step 5: Run predictions
    result = pred_model.predict(np.asarray(pred_df["name"].values.tolist())).squeeze(
        axis=1
    )

    # Step 6: Convert the probabilities to predictions
    pred_df["boy_or_girl"] = ["boy" if logit > 0.5 else "girl" for logit in result]
    pred_df["probability"] = [logit if logit > 0.5 else 1.0 - logit for logit in result]

    # Step 7: Format the output
    pred_df["name"] = names
    pred_df["probability"] = pred_df["probability"].round(2)
    pred_df.drop_duplicates(inplace=True)

    return {"response": pred_df.to_dict(orient="records")}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))