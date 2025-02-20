import cv2
import os 
import numpy as np 
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial import KDTree
import math 
import seaborn as sns 

def CreateDir(dir_path):
  if not os.path.exists(dir_path[:-1]):
     os.makedirs(dir_path[:-1])

def AdjustColorRange(image):
    norm_image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    return norm_image

def BackgroundSubtraction(image):
  background = cv2.GaussianBlur(image, (51, 51), 0)
  corrected_image = cv2.subtract(image, background)
  return corrected_image

def ThresholdWithWatershed(image):
      # Ensure the image is grayscale
    if len(image.shape) == 3:  # If the image has 3 channels
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply binary thresholding
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

    # Sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # Sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

    # Unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labelling
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    # Apply watershed
    markers = cv2.watershed(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), markers)

    # Mark boundaries in the original image
    segmented_image = image.copy()
    segmented_image[markers == -1] = 255  # Mark boundaries as white

    # Extract contours
    contours, _ = cv2.findContours(sure_fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return segmented_image, contours

def DataExtraction(data_path: str, i:int, contours, scalebar, min_contour_points = 10) -> dict:
  '''extracts shape metrics from each detected spot
  each contour contains a set of points defining the outline of a spot'''

  metrics = []
  filtered_contours = []

  if int(i) == 0:
      max_id = 0
  if int(i) != 0:
     max_id = None
  for contour in contours:
    #check enough points in countour, if theres not enough its probably too small
    if len(contour) >= min_contour_points:  
            perimeter = cv2. arcLength(contour, True)
            area = cv2.contourArea(contour)
            ellipse = cv2.fitEllipse(contour)
            (center, axes, orientation) = ellipse
            major_axis_length = max(axes)
            minor_axis_length = min(axes) 
    else:
       continue
 
    #filter out small spots
    if area * scalebar ** 2 > .25: #/mu m
        continue
    if perimeter * scalebar > 3: 
       continue
    if minor_axis_length * scalebar > .34:
        continue
    
    #get centroid metrics  
    M = cv2.moments(contour)
    if M['m00'] != 0:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
    else:
        cx, cy = None, None  
        continue
    
    #see if existing track
    spore_id, max_id = GetSporeID(data_path, [cx, cy], i, max_id)
    #if new spore, dont record
    if spore_id is None:
       continue
    
    #accepted contours
    filtered_contours.append(contour)
    #dictionary entry 
    metrics.append({
        'perimeter': perimeter,
        'area': area,
        'minor_axis_length': minor_axis_length,
        "x": cx, "y": cy,
        'image_idx': i,
        "spore_id": spore_id
    })
  return metrics, filtered_contours


def GetSporeID(data_path, centroid, i, max_id, xy_tol = 10) -> int:
#   might need to tell main data extraction to remove spot if spore id is not matched to time 0? to make postprocessing happen during imaging

  if int(i) == 0:
     if not os.path.exists(data_path):
        max_id += 1
        spore_id = max_id

     if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        max_id += 1
        spore_id = max_id 
     
  elif int(i) != 0:
     
     df = pd.read_csv(data_path)

     centroids_list = df[["x", "y"]].values  # Assumes columns are 'x' and 'y'
     kdtree = KDTree(centroids_list)
     matching_indices = kdtree.query_ball_point(centroid, xy_tol)
   #   print(f"found matching indices: {matching_indices}")
   # if found centroid
     if len(matching_indices) > 0:
        matched_id = df.iloc[matching_indices[0]]["spore_id"]  # Assumes column is 'spore_id'
        spore_id = matched_id
     else: 
        spore_id =  None
  return spore_id, max_id

def write_data(data, columns, i, output_csv):
       #WRITE DATA=================
    if i == 0:
      data.to_csv(output_csv, index=False, header = True)
    if i != 0:
      data.to_csv(output_csv, index=False, header = False, mode = "a")

def LineplotTrackFeature(data_input_path, lineplot_output_path, feature) -> None:
   df = pd.read_csv(data_input_path)
   spore_ids = df["spore_id"].unique()
   plt.figure(figsize = (6, 4))
   tracks = 0
   for spore_id in spore_ids:
      spore_df = df[df["spore_id"] == spore_id]
      sns.scatterplot(x = "image_idx", y = feature, data = spore_df,  s = 30)
      sns.lineplot(x = "image_idx", y = feature, data = spore_df,  linewidth = 1)
      tracks += 1
   plt.xlabel("Time index", fontsize = 18)
   plt.ylabel(feature.title(), fontsize = 18)
   plt.savefig(lineplot_output_path)
   print(f"plotting {tracks} spore {feature}s...")


def Gif(image_dir, gif_name, fps, output_dir):
    print(f"creating gif from {image_dir}...")
    # Get list of image files in the directory
    image_files = sorted(
        [os.path.join(image_dir, file) for file in os.listdir(image_dir) if file.endswith(('.png', '.jpg', '.jpeg'))]
    )
    if not image_files:
        raise FileNotFoundError(f"No valid image files found in {image_dir}")

    gif_path = os.path.join(output_dir, gif_name)
    
    # Create the GIF
    with imageio.get_writer(gif_path, mode='I', fps=fps) as writer:
        for image_file in image_files:
            image = imageio.imread(image_file)
            writer.append_data(image)
    
    print(f"gif saved as {gif_path}...")

def create_artif_data(metrics_t: list, prev_data_csv_path: str):
  df_t = pd.DataFrame(metrics_t)
  df_allt = pd.read_csv(prev_data_csv_path)
  t_idx = int(df_t["image_idx"].values[0])
  df_tprev = df_allt[df_allt["image_idx"] == t_idx - 1]
  sporeids_tprev = df_tprev["spore_id"].unique()
  sporeids_t = df_t["spore_id"].unique()
  missing_sporeids = [sporeid for sporeid in sporeids_tprev if sporeid not in sporeids_t]
  df_tprev_missing = df_tprev[df_tprev["spore_id"].isin(missing_sporeids)]
  df_t_artificial = df_tprev_missing.copy()
  df_t_artificial["image_idx"] = t_idx
  return df_t_artificial



##===============================MAIN===================================
if __name__ == "__main__":

  #PARAMETERS
  experiment = "M4581_s1"
  microscopy = "ThT"
  image_folder_path = "/Users/alexandranava/Desktop/Spores/Data/M4581_s1/ThT/"
  segmentation_repo = "/Users/alexandranava/Desktop/Spores/Spore_Segmentation/"
  scalebar = 0.065 #1px = 0.065/mu m
  
  original_image_dir = f"{segmentation_repo}OriginalImages/{experiment}_{microscopy}/"
  CreateDir(original_image_dir)
  #DIRS FOR PREPROCESS AND SEGMENTATION
  preprocessed_image_dir = f"{segmentation_repo}PreprocessedImages/{experiment}_{microscopy}/"
  CreateDir(preprocessed_image_dir)

  segm_output_csv = f"{segmentation_repo}SegmentedData_{experiment}_{microscopy}.csv"
  segmented_image_dir = f"{segmentation_repo}SegmentedImages/{experiment}_{microscopy}/"
  CreateDir(segmented_image_dir)


  #FOR IMAGES IN FOLDER ===================
  #for image_file in os.listdir(image_folder_path): # FOR EXP USE
  for test_idx in range(0, 100): #FOR TEST USE
    image_file = "M4581_s1_ThT_stabilized_" + str(test_idx).zfill(4) + ".tif" #FOR TEST USE
    image_path = os.path.join(image_folder_path, image_file)
    image_idx = image_file.split("_")[-1].replace(".tif", "")
    print(f"working on image {image_idx}...")
    image = cv2.imread(image_path)

    #ORIGINAL IMAGE
    plt.imshow(image)
    plt.axis('off')
    plt.savefig(os.path.join(original_image_dir, image_file.replace(".tif", ".jpg")), bbox_inches='tight', pad_inches=0)

    #PREPROCESSING==================
    image = BackgroundSubtraction(image)#background subtraction
    image = AdjustColorRange(image)#color range adjustment
    plt.imshow(image)
    plt.axis('off')
    plt.savefig(os.path.join(preprocessed_image_dir, image_file.replace(".tif", ".jpg")), bbox_inches='tight', pad_inches=0)
    #SEGMENTATION===================
    segmented_image, contours = ThresholdWithWatershed(image)#define contours
    plt.imshow(segmented_image, cmap = 'hot')
    plt.axis('off')

    #DATA EXTRACTION/SPOT FILTERING=================
    shape_metrics, filtered_contours = DataExtraction(segm_output_csv, image_idx, contours, scalebar)
    #df with spores data from last frame, but reindexed to current frame 
    #COMBINE TRUE AND ARTIFICIAL DATA
    cols = ["image_idx", "spore_id", "x", "y", "perimeter", "area", "minor_axis_length"]
    df = pd.DataFrame(shape_metrics, columns=cols)
    df["experimental"] = 1

    #POSTPROCESSING=================
    #create artificial data for missing spores from prev time points, removes track if more than a fraction of any cell track is artifical over window size on prev t 
    if test_idx > 0:
      artificial_data_df = create_artif_data(shape_metrics, segm_output_csv)
      artificial_data_df["experimental"] = 0
      df = pd.concat([df, artificial_data_df])

    #WRITE DATA=================
    write_data(df, cols, test_idx, segm_output_csv)
    #PLOT SEGMENTED IMAGE WITH SPOTS
    sns.scatterplot(x = "x", y = "y", data = df, color = "white", s=3)
    plt.savefig(os.path.join(segmented_image_dir, image_file.replace(".tif", ".jpg")), bbox_inches='tight', pad_inches=0)
    plt.close()
    
  #MAKE GIFS=========
  Gif(original_image_dir, f'{experiment}_{microscopy}_Unprocessed.gif', 3, segmentation_repo)
  Gif(preprocessed_image_dir, f'{experiment}_{microscopy}_Preprocessed.gif', 3, segmentation_repo)
  Gif(segmented_image_dir, f'{experiment}_{microscopy}_Segmented.gif', 3, segmentation_repo)


  #PLOTTING FEATURES===============
  #for feature in ["perimeter", "area", "minor_axis_length"]:
  for feature in ["area"]: 
    LineplotTrackFeature(segm_output_csv, f"{segmentation_repo + feature}_3tmp.jpg", feature)

