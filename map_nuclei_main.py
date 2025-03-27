import pandas as pd
from pathlib import Path
from scipy.spatial import cKDTree
from tqdm import tqdm

from compressed_sampler import SampleManager
import extract_coordinates_from_key as ecfk

#######################################################################################


def create_sample_df(filepath):
    pickle_filelist = [f for f in sorted(Path(filepath).glob(f"**/*_results"))]

    data_keys = list()
    data_centroid_x = list()
    data_centroid_y = list()

    for pickle_file in pickle_filelist:
        pickle_path = str(pickle_file)
        # print(wsi_path)

        nucmor_samples = SampleManager(filename=pickle_path)
        nucmor_samples.load_samples()

        # These are lists
        data_keys.extend(nucmor_samples.sample_keys)
        # I might actually not even need the centroid data -> might be faster, to just use the keys... need to check!
        data_sample_x = nucmor_samples.sample_xs
        for nucleus in data_sample_x:
            data_centroid_x.append(nucleus["centroid"][0])
            data_centroid_y.append(nucleus["centroid"][1])

    # Finally create a dataframe from all the data.
    df = pd.DataFrame(
        columns=["key", "x", "y"], data=zip(data_keys, data_centroid_x, data_centroid_y)
    )

    # execute all the functions from Indras script
    df["code"] = df["key"].apply(ecfk.get_code)
    df["xykey"] = df.apply(ecfk.make_xykey, axis=1)
    return df


def check_double_keys(df):
    # Testing, if two nuclei have the same key now
    if len(df["xykey"]) != len(df["xykey"].unique()):
        print(
            f"Warning: There are {len(df['xykey']) - len(df['xykey'].unique())} duplicate keys."
        )
        duplicates = df[df.duplicated(subset=["xykey"], keep=False)]
        value_counts = duplicates["xykey"].value_counts()
        duplication_counts = value_counts.value_counts().sort_index()
        print(duplication_counts)
    return None


#######################################################################################


def alternative_mapping(df1, df2, margin=3):
    result_df = pd.DataFrame()
    multiple_mappings = []

    for WSI in df2.code.unique():
        df1_subset = df1[df1.code == WSI]
        df2_subset = df2[df2.code == WSI]

        if df1_subset.empty or df2_subset.empty:
            print(f"No nuclei found for WSI {WSI}.")
            continue

        tree = cKDTree(df1_subset[["x", "y"]].values)
        distances, indices = tree.query(
            df2_subset[["x", "y"]].values, distance_upper_bound=margin
        )

        valid_matches = distances != float("inf")
        df2_subset = df2_subset[valid_matches].copy()
        df2_subset["mapped"] = df1_subset.iloc[indices[valid_matches]]["key"].values

        duplicated_keys = df2_subset["mapped"].duplicated(keep=False)
        if duplicated_keys.any():
            multiple_mappings.extend(df2_subset[duplicated_keys]["mapped"].unique())

        result_df = pd.concat([result_df, df2_subset])

    if multiple_mappings:
        print(
            f"Warning: Multiple entries in df2 were mapped to the same entry in df1. Len entries df1: {len(multiple_mappings)}"
        )
    print(f"Mapping done. {len(result_df)} nuclei were mapped.")
    return result_df


def _remove_double_mapped_nuclei(df):
    # Remove nuclei that are mapped to multiple nuclei in the other dataframe
    double_mapped = df["mapped"].duplicated(keep=False)
    if double_mapped.any():
        print(
            f"Warning: {double_mapped.sum()} nuclei were mapped to multiple nuclei in the other dataframe. Removing these."
        )
        df = df[~double_mapped]
    else:
        print("No double mapped nuclei found.")
    return df


#######################################################################################
# New function to combine all nuclei into one DataFrame
def combine_all_nuclei(unet_df, sam_05_df, sam_025_df, cvpp_256_df, margin=3):
    # Step 1: Prepare each DataFrame with its 'key' column assigned to the respective key column
    unet_df = unet_df[["code", "x", "y", "key"]].copy()
    unet_df["unet_key"] = unet_df["key"].astype(
        str
    )  # Ensure string type for consistency
    unet_df["sam_05_key"] = "NA"
    unet_df["sam_025_key"] = "NA"
    unet_df["cvpp_256_key"] = "NA"
    print(
        f"UNET loaded with {len(unet_df)} nuclei. Sample keys: {unet_df['unet_key'].head().tolist()}"
    )

    sam_05_df = sam_05_df[["code", "x", "y", "key"]].copy()
    sam_05_df["unet_key"] = "NA"
    sam_05_df["sam_05_key"] = sam_05_df["key"].astype(str)
    sam_05_df["sam_025_key"] = "NA"
    sam_05_df["cvpp_256_key"] = "NA"
    print(
        f"SAM_05 loaded with {len(sam_05_df)} nuclei. Sample keys: {sam_05_df['sam_05_key'].head().tolist()}"
    )

    sam_025_df = sam_025_df[["code", "x", "y", "key"]].copy()
    sam_025_df["unet_key"] = "NA"
    sam_025_df["sam_05_key"] = "NA"
    sam_025_df["sam_025_key"] = sam_025_df["key"].astype(str)
    sam_025_df["cvpp_256_key"] = "NA"
    print(
        f"SAM_025 loaded with {len(sam_025_df)} nuclei. Sample keys: {sam_025_df['sam_025_key'].head().tolist()}"
    )

    cvpp_256_df = cvpp_256_df[["code", "x", "y", "key"]].copy()
    cvpp_256_df["unet_key"] = "NA"
    cvpp_256_df["sam_05_key"] = "NA"
    cvpp_256_df["sam_025_key"] = "NA"
    cvpp_256_df["cvpp_256_key"] = cvpp_256_df["key"].astype(str)
    print(
        f"CVPP_256 loaded with {len(cvpp_256_df)} nuclei. Sample keys: {cvpp_256_df['cvpp_256_key'].head().tolist()}"
    )

    print("DFs created.")
    # Drop the original 'key' column since it's now distributed
    unet_df = unet_df.drop(columns=["key"])
    sam_05_df = sam_05_df.drop(columns=["key"])
    sam_025_df = sam_025_df.drop(columns=["key"])
    cvpp_256_df = cvpp_256_df.drop(columns=["key"])

    # Combine all into one DataFrame
    all_nuclei = pd.concat(
        [unet_df, sam_05_df, sam_025_df, cvpp_256_df], ignore_index=True
    )
    print("Combined.")

    # Step 2: Group by (code, x, y) to merge exact matches
    def merge_keys(series):
        # Take the first non-"NA" value in the series
        valid = series[series != "NA"]
        return valid.iloc[0] if not valid.empty else "NA"

    grouped = (
        all_nuclei.groupby(["code", "x", "y"])
        .agg(
            {
                "unet_key": merge_keys,
                "sam_05_key": merge_keys,
                "sam_025_key": merge_keys,
                "cvpp_256_key": merge_keys,
            }
        )
        .reset_index()
    )

    print("Keys merged and grouped.")

    # Step 3: Use cKDTree to merge nuclei within margin distance
    for WSI in tqdm(grouped["code"].unique()):
        subset = grouped[grouped["code"] == WSI]
        if len(subset) <= 1:
            continue

        tree = cKDTree(subset[["x", "y"]].values)
        pairs = tree.query_pairs(r=margin)

        to_drop = set()
        for i, j in pairs:
            idx_i, idx_j = subset.index[i], subset.index[j]
            if idx_i in to_drop or idx_j in to_drop:
                continue
            # Merge keys: keep non-"NA" value from either row
            for key_col in ["unet_key", "sam_05_key", "sam_025_key", "cvpp_256_key"]:
                if grouped.loc[idx_j, key_col] != "NA":
                    grouped.loc[idx_i, key_col] = grouped.loc[idx_j, key_col]
            to_drop.add(idx_j)

        grouped = grouped.drop(to_drop)

    print(f"Combined DataFrame created with {len(grouped)} unique nuclei.")
    print(f"Sample rows:\n{grouped.head()}")
    return grouped


#######################################################################################
if __name__ == "__main__":

    datapath = "../nucmor/results/H_and_E/"
    outpath = "../nucmor/morph_analysis/"
    UNET_nuclei_df = pd.read_csv(outpath + "UNET_211_subset.csv")

    # This is the old way of mapping the nuclei, which was a bit wrong. Below, you find a refined version.
    create_new = False
    if create_new:
        CVpp_SAM_05_df = create_sample_df(datapath + "CVpp_SAM_05_KTB/no_resize/")
        CVpp_SAM_025_df = create_sample_df(datapath + "CVpp_SAM_025_KTB/no_resize/")
        CVpp_256_05_df = create_sample_df(datapath + "CVpp_256_05_KTB/")
        print("Sample dfs created.")

        # save CVpp dataframes, so analysis can go quicker.
        CVpp_SAM_05_df.to_csv(datapath + "CVpp_SAM_05_KTB.csv", index=False)
        CVpp_SAM_025_df.to_csv(datapath + "CVpp_SAM_025_KTB.csv", index=False)
        CVpp_256_05_df.to_csv(datapath + "CVpp_256_05_KTB.csv", index=False)
        print("CVpp dataframes saved.")

        ## Commented out, as I saved the UNET_nuclei_df already with these properties.
        # UNET_nuclei_df[['x', 'y']] = UNET_nuclei_df['key'].apply(lambda x: pd.Series(ecfk.parse_coord_old(x)))
        UNET_nuclei_df["xykey"] = UNET_nuclei_df.apply(ecfk.make_xykey, axis=1)
        UNET_nuclei_reduc_df = UNET_nuclei_df[["code", "xykey"]].copy()
        print("Reduced UNET created")

        # Checking for double keys
        check_double_keys(CVpp_SAM_05_df)
        check_double_keys(CVpp_SAM_025_df)
        check_double_keys(CVpp_256_05_df)
        check_double_keys(UNET_nuclei_reduc_df)
        print("Double keys checked.")

    #  If you created the dataframes above, you can merge them here. Otherwise, you can also load the already created dataframes by setting the argument to true below.
    merge_new = False
    if merge_new:
        dataframes_exist = True
        if dataframes_exist:
            # Loading the already created dataframes from above -- if they are not created yet, please start the first if function.
            CVpp_SAM_05_df = pd.read_csv(outpath + "CVpp_SAM_05_KTB.csv")
            CVpp_SAM_025_df = pd.read_csv(outpath + "CVpp_SAM_025_KTB.csv")
            CVpp_256_05_df = pd.read_csv(outpath + "CVpp_256_05_KTB.csv")

            UNET_nuclei_df["xykey"] = UNET_nuclei_df.apply(ecfk.make_xykey, axis=1)
            UNET_nuclei_reduc_df = UNET_nuclei_df[["code", "xykey"]].copy()

        complete_dataset = pd.merge(
            UNET_nuclei_reduc_df,
            CVpp_SAM_05_df[["xykey", "key"]],
            on="xykey",
            how="left",
        )
        complete_dataset = complete_dataset.rename(columns={"key": "CVpp_SAM_05"})
        print("Merged CVpp_SAM_05")

        complete_dataset = pd.merge(
            complete_dataset, CVpp_SAM_025_df[["xykey", "key"]], on="xykey", how="left"
        )
        complete_dataset = complete_dataset.rename(columns={"key": "CVpp_SAM_025"})
        print("Merged CVpp_SAM_025")

        complete_dataset = pd.merge(
            complete_dataset, CVpp_256_05_df[["xykey", "key"]], on="xykey", how="left"
        )
        complete_dataset = complete_dataset.rename(columns={"key": "CVpp_256_05"})
        print("Merged CVpp_256_05")

        complete_dataset.to_csv("mapped_nuclei_KTB_UNet_CVpp_px10.csv", index=False)
        print("Merged dataset saved.")

    # New mapping function execution
    start_alt_mapping = False
    if start_alt_mapping:
        margin_radius = 3
        restart_mapping = False
        if restart_mapping:
            # Loading the already created dataframes from above -- if they are not created yet, please start the first if function.
            CVpp_SAM_05_df = pd.read_csv(outpath + "CVpp_SAM_05_KTB.csv")
            CVpp_SAM_025_df = pd.read_csv(outpath + "CVpp_SAM_025_KTB.csv")
            CVpp_256_05_df = pd.read_csv(outpath + "CVpp_256_05_KTB.csv")
            print("Dataframes loaded. Starting mapping.")

            # Execute the new mapping function
            CVpp_SAM_05_mapped_df = alternative_mapping(
                UNET_nuclei_df, CVpp_SAM_05_df, margin=margin_radius
            )
            CVpp_SAM_025_mapped_df = alternative_mapping(
                UNET_nuclei_df, CVpp_SAM_025_df, margin=margin_radius
            )
            CVpp_256_05_mapped_df = alternative_mapping(
                UNET_nuclei_df, CVpp_256_05_df, margin=margin_radius
            )
            print("Alternative mapping done.")

            # Removing double mapped nuclei if desired.
            remove_double_mapped = True
            if remove_double_mapped:
                CVpp_SAM_05_mapped_df = _remove_double_mapped_nuclei(
                    CVpp_SAM_05_mapped_df
                )
                CVpp_SAM_025_mapped_df = _remove_double_mapped_nuclei(
                    CVpp_SAM_025_mapped_df
                )
                CVpp_256_05_mapped_df = _remove_double_mapped_nuclei(
                    CVpp_256_05_mapped_df
                )
                print("Double mapped nuclei removed.")

            # Save as csv to the outdirectory.
            save_as_csv = False
            if save_as_csv:
                CVpp_SAM_05_mapped_df.to_csv(
                    outpath + f"CVpp_SAM_05_r{margin_radius}_KTB_mapped.csv",
                    index=False,
                )
                CVpp_SAM_025_mapped_df.to_csv(
                    outpath + f"CVpp_SAM_025_r{margin_radius}_KTB_mapped.csv",
                    index=False,
                )
                CVpp_256_05_mapped_df.to_csv(
                    outpath + f"CVpp_256_05_r{margin_radius}_KTB_mapped.csv",
                    index=False,
                )
                print("Mapped dataframes saved.")

        # if the analysis above was already done, you can just load the dataframes here
        load_mapped_dfs = False
        if load_mapped_dfs:
            CVpp_SAM_05_mapped_df = pd.read_csv(
                outpath + "CVpp_SAM_05_r3_KTB_mapped.csv"
            )
            CVpp_SAM_025_mapped_df = pd.read_csv(
                outpath + "CVpp_SAM_025_r3_KTB_mapped.csv"
            )
            CVpp_256_05_mapped_df = pd.read_csv(
                outpath + "CVpp_256_05_r3_KTB_mapped.csv"
            )
            print("Mapped dataframes loaded.")

        # Create one big df with all the mapped data
        create_all_mapped = False
        if create_all_mapped:
            complete_mapped_df = pd.merge(
                UNET_nuclei_df[["key", "code", "x", "y"]],
                CVpp_SAM_05_mapped_df[["key", "mapped"]],
                left_on="key",
                right_on="mapped",
                how="left",
            )
            complete_mapped_df = complete_mapped_df.rename(
                columns={"key_x": "UNET_key", "key_y": "CVpp_SAM_05"}
            ).drop(columns=["mapped"])
            complete_mapped_df = pd.merge(
                complete_mapped_df,
                CVpp_SAM_025_mapped_df[["key", "mapped"]],
                left_on="UNET_key",
                right_on="mapped",
                how="left",
            )
            complete_mapped_df = complete_mapped_df.rename(
                columns={"key": "CVpp_SAM_025"}
            ).drop(columns=["mapped"])
            complete_mapped_df = pd.merge(
                complete_mapped_df,
                CVpp_256_05_mapped_df[["key", "mapped"]],
                left_on="UNET_key",
                right_on="mapped",
                how="left",
            )
            complete_mapped_df = complete_mapped_df.rename(
                columns={"key": "CVpp_256_05"}
            ).drop(columns=["mapped"])

            complete_mapped_df.to_csv(
                outpath + f"new_mapped_nuclei_KTB_UNet_CVpp_r{margin_radius}.csv",
                index=False,
            )
            print("New complete mapped dataframe saved.")

        # Create a mapped dataframe between the three CV++ models.
        create_CVpp_mapped = False
        if create_CVpp_mapped:
            print("Creating CVPP mappings")

    map_between_all_tools = True
    if map_between_all_tools:
        # Load the DataFrames
        CVpp_SAM_05_df = pd.read_csv(outpath + "CVpp_SAM_05_KTB.csv")
        CVpp_SAM_025_df = pd.read_csv(outpath + "CVpp_SAM_025_KTB.csv")
        CVpp_256_05_df = pd.read_csv(outpath + "CVpp_256_05_KTB.csv")
        print("Dataframes loaded.")

        # Combine all nuclei into one DataFrame
        margin_radius = 3
        all_nuclei_df = combine_all_nuclei(
            UNET_nuclei_df,
            CVpp_SAM_05_df,
            CVpp_SAM_025_df,
            CVpp_256_05_df,
            margin=margin_radius,
        )

        # Save the result
        save_as_csv = True
        if save_as_csv:
            all_nuclei_df.to_csv(
                outpath + f"all_nuclei_r{margin_radius}_combined_with-keys.csv",
                index=False,
            )
            print("Combined DataFrame saved.")

    print("Done!")
