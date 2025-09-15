import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from common import print_machine_info, benchmark_pipeline

df = pd.read_csv("combined.csv")

X = df.drop(columns=["Temp"])
y = df["Temp"]

num_features = ["RelH", "L1", "L2"]   # continuous -> scale
cat_features = ["Act", "room"]        # categorical -> one-hot
pass_features = ["Occ", "Door", "Win"]  # passthrough (already numeric/binary)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_features),
        ("passthrough", "passthrough", pass_features),
    ],
    remainder="drop"
)

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

print_machine_info()

metrics = benchmark_pipeline(model, X_train, y_train, X_test, y_test)

try:
    combined_test = pd.concat([
        X_test.reset_index(drop=True),
        y_test.reset_index(drop=True).rename("Temp")
    ], axis=1)
    combined_test["y_pred"] = metrics["y_pred"]

    per_room = combined_test.groupby("room", group_keys=False).apply(
    lambda df: np.sqrt(mean_squared_error(df["Temp"], df["y_pred"])),
    include_groups=False).rename("rmse").to_frame()

    print("\nPer-room RMSE:")
    print(per_room)
except Exception as e:
    print("Per-room RMSE skipped (error):", e)
