import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


def create_clean_csv(demo_path, rec_path, included_ids, out_path):

    demo = pd.read_csv(demo_path, sep=';')
    rec = pd.read_csv(rec_path, sep=';')

    # ---------- Keep only included MRI scans ----------
    demo = demo[demo["MRI_ID"].isin(included_ids)]
    rec = rec[rec["MRI_ID"].isin(included_ids)]

    # ---------- Merge correctly on MRI_ID ----------
    df = pd.merge(demo, rec, on="MRI_ID", how="inner")

    # ---------- Remove duplicate rows ----------
    df = df.drop_duplicates(subset="MRI_ID")

    # ---------- Select relevant columns only ----------
    keep_cols = [
        "MRI_ID",
        "diagnosis",
        "sex",
        "Age",
        "years_education",
        "MMSE",
        "MRIAcquisitionType",
        "RepetitionTime",
        "ImagedNucleus",
        "MagneticFieldStrength"
    ]

    df = df[[c for c in keep_cols if c in df.columns]]

         # ✅ FIX 1 — safe numeric conversion (NEW pandas compatible)
    for c in df.columns:
        if c != "MRI_ID":
            converted = pd.to_numeric(df[c], errors="coerce")
            if converted.notna().sum() > 0:
                df[c] = converted

    # ✅ FIX 2 — safe missing value handling
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].fillna(df[c].median())
        else:
            df[c] = df[c].fillna("Unknown")

    # ---------- Normalize Age ----------
    if "Age" in df.columns:
        df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
        df["Age"] = df["Age"].fillna(df["Age"].median())

        scaler = MinMaxScaler()
        df[["Age"]] = scaler.fit_transform(df[["Age"]])

    # ---------- One-hot encode categorical ----------
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    cat_cols.remove("MRI_ID")  # keep ID as identifier

    if len(cat_cols) > 0:
        enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        enc_data = enc.fit_transform(df[cat_cols])

        enc_df = pd.DataFrame(
            enc_data,
            columns=enc.get_feature_names_out(cat_cols),
            index=df.index
        )

        df = pd.concat([df.drop(columns=cat_cols), enc_df], axis=1)

    # ---------- Save ----------
    df.to_csv(out_path, index=False)

    print("Saved cleaned CSV:", out_path)
    print("Final shape:", df.shape)
