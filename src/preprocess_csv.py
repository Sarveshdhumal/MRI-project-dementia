import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

def create_clean_csv(demo_path, rec_path, included_ids, out_path):

    demo = pd.read_csv(demo_path, sep=';')
    rec = pd.read_csv(rec_path, sep=';')

   
    demo = demo.reset_index(drop=True)
    rec = rec.reset_index(drop=True)

    df = pd.concat([demo, rec], axis=1)

   
    df = df.loc[:, ~df.columns.duplicated()]

    print("Merged shape:", df.shape)

   
    if included_ids:
        mask = df.apply(
            lambda row: any(str(i) in str(row.values) for i in included_ids),
            axis=1
        )
        df = df[mask]

    
    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = df[c].fillna("Unknown")
        else:
            df[c] = df[c].fillna(df[c].median())

    
    from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

   
    if "Age" in df.columns:
        df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
        scaler = MinMaxScaler()
        df[["Age"]] = scaler.fit_transform(df[["Age"]])


    
    cat_cols = df.select_dtypes(include="object").columns
    if len(cat_cols) > 0:
        enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        enc_data = enc.fit_transform(df[cat_cols])
        enc_df = pd.DataFrame(enc_data, columns=enc.get_feature_names_out(cat_cols))
        df = pd.concat([df.drop(columns=cat_cols), enc_df], axis=1)

    df.to_csv(out_path, index=False)

    print("Saved:", out_path)
