import numpy as np
import geopandas as gpd
import pandas as pd
import laspy
from shapely import wkt
from shapely.wkt import loads
import re
from shapely.geometry import box, mapping
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils.multiclass import unique_labels
import rasterio
from rasterio.mask import mask
from rasterio.transform import array_bounds
from rasterstats import zonal_stats
from tqdm import tqdm
import gc
import os

# helper functions
# functions for feature calculation
# vegetation indices
def calculate_devi(red, green, blue):
    devi = (green / (3 * green)) + (red / (3 * green)) + (blue / (3 * green)) 
    mean_devi = np.nanmean(devi)
    return mean_devi
def calculate_rgbvi(red, green, blue):
    rgbvi = ((green**2) - (blue * red)) / ((green**2) + (blue * red))
    mean_rgbvi = np.nanmean(rgbvi)
    return mean_rgbvi
    
def calculate_egrbdi(red, green, blue):
    egrbdi = (((2*green)**2) - (blue * red)) / (((2*green)**2) + (blue * red))
    mean_egrbdi = np.nanmean(egrbdi)
    return mean_egrbdi

def calculate_gli(red, green, blue):
    gli = ((2*green)-blue-red) / ((2*green)+blue+red) 
    mean_gli = np.nanmean(gli)
    return mean_gli

def calculate_exg(red, green, blue):
    exg = (2*green) - red
    mean_exg = np.nanmean(exg)
    return mean_exg

def calculate_exr(red, green, blue):
    exr = (1.4*red) - green
    mean_exr = np.nanmean(exr)
    return mean_exr

def calculate_vari(red, green, blue):
    vari = (green - red) / (green + red - blue)
    mean_vari = np.nanmean(vari)
    return mean_vari

# rgb stuff
def calculate_red(red):
    mean_red = np.nanmean(red)
    return mean_red
def calculate_green(green):
    mean_green = np.nanmean(green)
    return mean_green
def calculate_blue(blue):
    mean_blue = np.nanmean(blue)
    return mean_blue
def calculate_mean_rgb(red, green, blue):
    mean_rgb = np.nanmean([red, green, blue])
    return mean_rgb
def calculate_total_rgb(red, green, blue):
    total_rgb = np.nansum([red, green, blue])
    return total_rgb
    
def calculate_mean_intensity(intensity):
    mean_intensity = np.mean(intensity)
    return mean_intensity   
def calculate_std_intensity(intensity):
    std_intensity = np.std(intensity)
    return std_intensity
def calculate_p95_intensity(intensity):
    p95_intensity = np.percentile(intensity, 95)
    return p95_intensity
def calculate_mean_number_of_returns(number_of_returns):
    mean_number_of_returns = np.mean(number_of_returns)
    return mean_number_of_returns

def parse_geometry(geom_str):
    # Remove SRID and parse the WKT
    try:
        if isinstance(geom_str, str) and geom_str.startswith("SRID="):
            wkt_str = re.sub(r"SRID=\d+;", "", geom_str)
            return loads(wkt_str)
        return None
    except:
        return None




pr_list = []
def extract_all(las_path, all_crowns_gpkg_path, tif_path):
    las = laspy.read(las_path)
    all_crowns_gpkg = gpd.read_file(all_crowns_gpkg_path, layer="hulls")
    public_trees = pd.read_csv("results/bomen-stamgegevens-2025-06-17T13_25_16.778118+02_00.csv")

    print("Matching points to unique TreeID...")
    unique_tree_ids = np.unique(las.treeID)
    feature_rows = []  # ← this initializes the list

    for tree_id in tqdm(unique_tree_ids, desc="Processing TreeIDs"):
        indices = np.where(las.treeID == tree_id)[0]
        pts = las[indices]

        red = pts.red
        green = pts.green
        blue = pts.blue
        intensity = pts.intensity
        number_of_returns = pts.number_of_returns

        feature_row = {
            "treeID": tree_id,
            "DEVI": calculate_devi(red, green, blue),
            "RGBVI": calculate_rgbvi(red, green, blue),
            "EGRBDI": calculate_egrbdi(red, green, blue),
            "GLI": calculate_gli(red, green, blue),
            "EXG": calculate_exg(red, green, blue),
            "EXR": calculate_exr(red, green, blue),
            "VARI": calculate_vari(red, green, blue),
            "RED": calculate_red(red),
            "GREEN": calculate_green(green),
            "BLUE": calculate_blue(blue),
            "mean_intensity": calculate_mean_intensity(intensity),
            "std_intensity": calculate_std_intensity(intensity),
            "p95_intensity": calculate_p95_intensity(intensity),
            "mean_number_of_returns": calculate_mean_number_of_returns(number_of_returns)
        }

        feature_rows.append(feature_row)
    df_indices = pd.DataFrame(feature_rows)
    # Create a dictionary to store points per tree
    # points_per_tree = {}
    # for tree_id in tqdm(unique_tree_ids, desc="Processing TreeIDs"):
    #     indices = np.where(las.treeID == tree_id)[0]
    #     points_per_tree[tree_id] = las[indices]

    # print("Done matching points.")

    # print("Calculating first set of features per TreeID...")
    # # calculate features per tree
    # devi_per_tree = {}
    # rgbvi_per_tree = {}
    # egrbdi_per_tree = {}
    # gli_per_tree = {}
    # exg_per_tree = {}
    # exr_per_tree = {}
    # vari_per_tree = {}
    # red_per_tree = {}
    # green_per_tree = {}
    # blue_per_tree = {}
    # mean_intensity_per_tree = {}
    # std_intensity_per_tree = {}
    # p95_intensity_per_tree = {}
    # mean_number_of_returns_per_tree = {}

    # for tree_id in points_per_tree:
    #     red = points_per_tree[tree_id].red
    #     green = points_per_tree[tree_id].green
    #     blue = points_per_tree[tree_id].blue
    #     intensity = points_per_tree[tree_id].intensity
    #     number_of_returns = points_per_tree[tree_id].number_of_returns
        
    #     devi_per_tree[tree_id] = calculate_devi(red, green, blue)
    #     rgbvi_per_tree[tree_id] = calculate_rgbvi(red, green, blue)
    #     egrbdi_per_tree[tree_id] = calculate_egrbdi(red, green, blue)
    #     gli_per_tree[tree_id] = calculate_gli(red, green, blue)
    #     exg_per_tree[tree_id] = calculate_exg(red, green, blue)
    #     exr_per_tree[tree_id] = calculate_exr(red, green, blue)
    #     vari_per_tree[tree_id] = calculate_vari(red, green, blue)
    #     red_per_tree[tree_id] = calculate_red(red)
    #     green_per_tree[tree_id] = calculate_green(green)
    #     blue_per_tree[tree_id] = calculate_blue(blue)
    #     mean_intensity_per_tree[tree_id] = calculate_mean_intensity(intensity)
    #     std_intensity_per_tree[tree_id] = calculate_std_intensity(intensity)
    #     p95_intensity_per_tree[tree_id] = calculate_p95_intensity(intensity)
    #     mean_number_of_returns_per_tree[tree_id] = calculate_mean_number_of_returns(number_of_returns)
    # print("Done calculating first set of features.")

    public_trees['geometry'] = public_trees['Geometrie'].apply(parse_geometry)
    public_trees = gpd.GeoDataFrame(public_trees, geometry='geometry', crs="EPSG:28992")
    
    # print("Merge features into dataframe...")
    # # add features to dataframe
    # df_devi = pd.DataFrame(list(devi_per_tree.items()), columns=["treeID", "DEVI"])
    # df_rgbvi = pd.DataFrame(list(rgbvi_per_tree.items()), columns=["treeID", "RGBVI"])
    # df_egrbdi = pd.DataFrame(list(egrbdi_per_tree.items()), columns=["treeID", "EGRBDI"])
    # df_gli = pd.DataFrame(list(gli_per_tree.items()), columns=["treeID", "GLI"])
    # df_exg = pd.DataFrame(list(exg_per_tree.items()), columns=["treeID", "EXG"])
    # df_exr = pd.DataFrame(list(exr_per_tree.items()), columns=["treeID", "EXR"])
    # df_vari = pd.DataFrame(list(vari_per_tree.items()), columns=["treeID", "VARI"])
    # df_red = pd.DataFrame(list(red_per_tree.items()), columns=["treeID", "RED"])
    # df_green = pd.DataFrame(list(green_per_tree.items()), columns=["treeID", "GREEN"])
    # df_blue = pd.DataFrame(list(blue_per_tree.items()), columns=["treeID", "BLUE"])
    # df_mean_intensity = pd.DataFrame(list(mean_intensity_per_tree.items()), columns=["treeID", "mean_intensity"])
    # df_std_intensity = pd.DataFrame(list(std_intensity_per_tree.items()), columns=["treeID", "std_intensity"])
    # df_p95_intensity = pd.DataFrame(list(p95_intensity_per_tree.items()), columns=["treeID", "p95_intensity"])
    # df_mean_number_of_returns = pd.DataFrame(list(mean_number_of_returns_per_tree.items()), columns=["treeID", "mean_number_of_returns"])

    # df_indices = df_devi.merge(df_rgbvi, on="treeID", how="outer")
    # df_indices = df_indices.merge(df_gli, on="treeID", how="outer")
    # df_indices = df_indices.merge(df_egrbdi, on="treeID", how="outer")
    # df_indices = df_indices.merge(df_exg, on="treeID", how="outer")
    # df_indices = df_indices.merge(df_exr, on="treeID", how="outer")
    # df_indices = df_indices.merge(df_vari, on="treeID", how="outer")
    # df_indices = df_indices.merge(df_red, on="treeID", how="outer")
    # df_indices = df_indices.merge(df_green, on="treeID", how="outer")
    # df_indices = df_indices.merge(df_blue, on="treeID", how="outer")
    # df_indices = df_indices.merge(df_mean_intensity, on="treeID", how="outer")
    # df_indices = df_indices.merge(df_std_intensity, on="treeID", how="outer")
    # df_indices = df_indices.merge(df_p95_intensity, on="treeID", how="outer")
    # df_indices = df_indices.merge(df_mean_number_of_returns, on="treeID", how="outer")

    gdf_combined_all = all_crowns_gpkg.merge(df_indices, on="treeID", how="left")
    print("Done merging dataframe.")
    # clean up public trees dataset
    public_trees = public_trees.drop(['ID'], axis=1)
    public_trees = public_trees.rename(columns={'Beheerobject wetenschappelijke naam': 'genus'})

    print("Matching public trees to crowns...")
    # match public trees to crowns
    # Ensure both GeoDataFrames are in the same CRS
    public_trees = public_trees.to_crs(gdf_combined_all.crs)

    # Spatial join: find nearest point (public tree) to each crown polygon
    gdf_matched = gpd.sjoin(
        gdf_combined_all,
        public_trees,
        how="left",
        predicate="contains"
    )

    print("Done matching trees to crowns.")

    # remove duplicates 
    print("Removing duplicates...")
    # Step 3: For each crown (treeID), get the most common genus from matching public tree points
    genus_mode = gdf_matched.groupby("treeID")["genus"].agg(
        lambda x: x.mode().iloc[0] if not x.mode().empty else None
    )

    # Step 4: Map the result back to the crown GeoDataFrame
    gdf_combined_all["genus"] = gdf_combined_all["treeID"].map(genus_mode)
    print("Done removing duplicates.")

    # adding last features
    print("Adding more features...")
    # Crown depth
    gdf_combined_all['crown_depth'] = gdf_combined_all['zq95'] - gdf_combined_all['zq10']

    # Canopy slenderness (avoid division by zero)
    gdf_combined_all['canopy_slenderness'] = gdf_combined_all['zmax'] / np.sqrt(gdf_combined_all['area'].replace(0, np.nan))

    # Point density (avoid division by zero)
    gdf_combined_all['point_density'] = gdf_combined_all['n'] / gdf_combined_all['area'].replace(0, np.nan)

    print("Done adding more features.")

    missing = gdf_combined_all["genus"].isna().sum()
    total = len(gdf_combined_all)
    percentage_o = missing/total
    pr = round((percentage_o * 100),1)
    pr_list.append(pr)

    # --- Read NDVI raster ---
    with rasterio.open(tif_path) as src:
        print("Calculating NDVI...")
        red = src.read(2).astype("float32")
        nir = src.read(1).astype("float32")
        ndvi = (nir - red) / (nir + red + 1e-6)
        profile = src.profile.copy()
        profile.update(dtype="float32", count=1)
        print("NDVI calculated.")
    gdf_combined_all = gdf_combined_all.to_crs(src.crs)

    with rasterio.io.MemoryFile() as memfile:
        with memfile.open(**profile) as ndvi_ds:
            ndvi_ds.write(ndvi, 1)

            print("Computing zonal stats for NDVI...")
            ndvi_stats = zonal_stats(
                gdf_combined_all.geometry,
                ndvi_ds.read(1),
                affine=ndvi_ds.transform,
                stats=["mean"],
                nodata=None
            )
            gdf_combined_all["ndvi_mean"] = [s["mean"] for s in ndvi_stats]

            # Compute vegetation masks only once
            for thr in [0.1]:
                mask_arr = (ndvi > thr).astype("float32")

                print(f"Computing veg_frac_gt_{thr}...")
                stats = zonal_stats(
                    gdf_combined_all.geometry,
                    mask_arr,
                    affine=ndvi_ds.transform,
                    stats=["mean"],
                    nodata=None
                )
                gdf_combined_all[f"veg_frac_gt_{thr}"] = [s["mean"] for s in stats]
    print("Done calculating masks.")
    print("Filtering gdf...")
    gdf_clean = gdf_combined_all[
        (gdf_combined_all["ndvi_mean"] > 0.1) &
        (gdf_combined_all["veg_frac_gt_0.1"] > 0.2)
    ]
    print("Done filtering gdf.")
    print("Saving gdf_clean_{}.gpkg".format(tile_name))
    gdf_clean.to_file('cleaned_data_25GN1/gdf_clean_{}.gpkg'.format(tile_name))



tif_path = "CIR_25GN1_lzw.tif"
output_dir = 'cleaned_data_25GN1'
os.makedirs(output_dir, exist_ok=True)

tile_list = ["25GN1_01", "25GN1_02", "25GN1_03", "25GN1_04", "25GN1_05", "25GN1_06"]
for tile_name in tile_list:
    output_path = os.path.join(output_dir, f"gdf_clean_{tile_name}.gpkg")
    if os.path.exists(output_path):
        print(f"Skipping {tile_name} — cleaned output already exists.")
        continue
    print("Processing {}".format(tile_name))
    extract_all("25GN1_out/{}_trees.laz".format(tile_name), "25GN1_out/{}.gpkg".format(tile_name), tif_path)
    print("Done processing {}".format(tile_name))
    gc.collect()

df_pr = pd.DataFrame(pr_list)
df_pr.to_csv('results/gdf_clean_metadata_25GN1.csv')
