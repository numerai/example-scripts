from lightgbm import LGBMRegressor
import sys
import gc
import json
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import cloudpickle
import scipy

from numerapi import NumerAPI

ERA_COL = "era"
TARGET_COL = "target_cyrus_v4_20"
DATA_TYPE_COL = "data_type"
MODEL_FOLDER = "models"

def load_model(name):
    path = Path(f"{MODEL_FOLDER}/{name}.pkl")
    if path.is_file():
        model = pd.read_pickle(f"{MODEL_FOLDER}/{name}.pkl")
    else:
        model = False
    return model

def neutralize(
    df, columns, neutralizers=None, proportion=1.0, normalize=True, era_col="era", verbose=False
):
    if neutralizers is None:
        neutralizers = []
    unique_eras = df[era_col].unique()
    computed = []
    if verbose:
        iterator = tqdm(unique_eras)
    else:
        iterator = unique_eras
    for u in iterator:
        df_era = df[df[era_col] == u]
        scores = df_era[columns].values
        if normalize:
            scores2 = []
            for x in scores.T:
                x = (scipy.stats.rankdata(x, method="ordinal") - 0.5) / len(x)
                x = scipy.stats.norm.ppf(x)
                scores2.append(x)
            scores = np.array(scores2).T
        exposures = df_era[neutralizers].values

        scores -= proportion * exposures.dot(
            np.linalg.pinv(exposures.astype(np.float32), rcond=1e-6).dot(
                scores.astype(np.float32)
            )
        )

        scores /= scores.std(ddof=0)

        computed.append(scores)

    return pd.DataFrame(np.concatenate(computed), columns=columns, index=df.index)

napi = NumerAPI()

current_round = napi.get_current_round()
dataset_name = "v4.1"
feature_set_name = "small"

Path(f"./{dataset_name}").mkdir(parents=False, exist_ok=True)

napi.download_dataset(f"{dataset_name}/features.json")
with open(f"{dataset_name}/features.json", "r") as f:
    feature_metadata = json.load(f)
features = feature_metadata["feature_sets"][
    feature_set_name
] 

print('Reading live data')
target_cols = feature_metadata["targets"]
read_columns = features + target_cols + [ERA_COL, DATA_TYPE_COL]
dataset_file = f"{dataset_name}/live_int8_{current_round}.parquet"
print(dataset_file)
live_data = pd.read_parquet(dataset_file, columns=read_columns)

targets = [
    "target_nomi_v4_20",
    "target_jerome_v4_60",
    "target_ralph_v4_20",
    "target_tyler_v4_20",
    "target_victor_v4_20",
    "target_waldo_v4_20",
]
models = {}

params_name = f"lg_lgbm"

for target in targets:
    prediction_col = f"{params_name}_{dataset_name}_{feature_set_name}_{target}"
    all_data_model_name = f"all_data_{prediction_col}"
    print(f"Loading model '{all_data_model_name}'")
    models[prediction_col] = load_model(all_data_model_name)

def predict(live_data):
    live_data[features] = live_data[features].astype("int8") 
    for prediction_col, model in models.items():
        print(prediction_col, model)
        live_data[prediction_col] = model.predict(
            live_data[features].fillna(np.nan)
        )  
        gc.collect()

    prediction_cols = list(models.keys())
    print(f"Building ensemble with {prediction_cols}")
    live_data["equal_weight"] = live_data[prediction_cols].mean(axis=1)

    prediction_cols.append("equal_weight")

    print("Neutralizing")
    live_data["half_neutral_equal_weight"] = neutralize(
        df=live_data,
        columns=[f"equal_weight"],
        neutralizers=features,
        proportion=0.5,
        normalize=True,
        era_col=ERA_COL,
        verbose=True,
    )

    prediction_cols.append("half_neutral_equal_weight")

    model_to_submit = f"half_neutral_equal_weight"
    live_data["prediction"] = live_data[model_to_submit].rank(pct=True)
    return live_data.to_frame("prediction")

# test predict method
predictions = predict(live_data)
predictions.to_csv(f"live_predictions_{current_round}.csv")

# export pickle
p = cloudpickle.dumps(predict)
model_pkl = f"sunshine_{'_'.join(sys.version.split('.')[:2])}.pkl"
print(f"Saving model to {model_pkl}")
with open(model_pkl, "wb") as f:
        f.write(p)

