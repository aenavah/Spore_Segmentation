import cv2
import os 
import numpy as np 
import imageio
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

def Threshold(image):
  '''segment the image based on detected edges'''
  edges = cv2.Canny(image, 100, 200)
  contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  filled_image = np.zeros_like(image)
  cv2.drawContours(filled_image, contours, -1, 255, thickness=cv2.FILLED)
  return filled_image, contours

def DataExtraction(data_path: str, i:int, contours, min_contour_points = 10) -> dict:
  '''extracts shape metrics from each detected spot
  each contour contains a set of points defining the outline of a spot'''
  metrics = []
  if int(i) == 0:
      max_id = 0
  if int(i) != 0:
     max_id = None
  for contour in contours:
    if len(contour) >= min_contour_points:  
            perimeter = cv2. arcLength(contour, True)
            area = cv2.contourArea(contour)
            ellipse = cv2.fitEllipse(contour)
            (center, axes, orientation) = ellipse
            major_axis_length = max(axes)
            minor_axis_length = min(axes) 
    else:
       continue
    if area<200:
       continue
    if area>400:
       continue
    if perimeter > 100:
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
    if spore_id == None:
       continue
    #write data
    metrics.append({
        'perimeter': perimeter,
        'area': area,
        'minor_axis_length': minor_axis_length,
        "x": cx, "y": cy,
        'image_idx': i,
        "spore_id": spore_id
    })
  return metrics


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
    data = pd.DataFrame(data, columns=columns)
    if i == 0:
      data.to_csv(output_csv, index=False, header = True)
    if i != 0:
      data.to_csv(output_csv, index=False, header = False, mode = "a")


def FilterbyTrackStart(data_input_path, data_output_path, frame_idx_criteria) -> None:
   print(f"filtering data by track start...")
   df = pd.read_csv(data_input_path)

   filtered_data = []
   filtered_count = 0
   accepted_count = 0

   unique_spore_ids = df["spore_id"].unique()
   for spore_id in unique_spore_ids:
      spore_df = df[df["spore_id"]== spore_id]
      if frame_idx_criteria in spore_df["image_idx"].values:
          filtered_data.append(spore_df)
          accepted_count += 1
      else:
         filtered_count += 1

   if filtered_data:
      filtered_df = pd.concat(filtered_data)
      filtered_df.to_csv(data_output_path, index=False)
      print(f"\t filtered {filtered_count} tracks...")
      print(f"\t accepted {accepted_count} tracks...")
   else:
      print("\t no tracks meet the specified criteria...")

def FilterbyTrackLength(data_input_path, data_output_path, length_threshold) -> None:
   print(f"filtering tracks by length...")
   df = pd.read_csv(data_input_path)

   filtered_data = []
   filtered_count = 0
   accepted_count = 0

   unique_spore_ids = df["spore_id"].unique()
   for spore_id in unique_spore_ids:
      spore_df = df[df["spore_id"]== spore_id]
      if len(spore_df) >= length_threshold:
         filtered_data.append(spore_df)
         accepted_count += 1
      else:
         filtered_count += 1
   if filtered_data:
      filtered_df = pd.concat(filtered_data)
      filtered_df.to_csv(data_output_path, index=False)
      print(f"\t filtered {filtered_count} tracks...")
      print(f"\t accepted {accepted_count} tracks...")
   else:
      print("\t no tracks meet the specified criteria...")

def LineplotTrackFeature(data_input_path, lineplot_output_path, feature) -> None:
   df = pd.read_csv(data_input_path)
   spore_ids = df["spore_id"].unique()
   plt.figure(figsize = (6, 5))
   for spore_id in spore_ids:
      spore_df = df[df["spore_id"] == spore_id]
      sns.lineplot(x = "image_idx", y = feature, data = spore_df)

   plt.savefig(lineplot_output_path)
   print(f"plotting {len(spore_ids)} spore {feature}s...")


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


##===============================MAIN===================================
if __name__ == "__main__":

  #PARAMETERS
  experiment = "M4581_s1"
  microscopy = "ThT"
  image_folder_path = "/Users/alexandranava/Desktop/Spores/Data/M4581_s1/ThT/"
  segmentation_repo = "/Users/alexandranava/Desktop/Spores/Spore_Segmentation/"
  
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
  for test_idx in range(0, 60): #FOR TEST USE
    image_file = "M4581_s1_ThT_stabilized_" + str(test_idx).zfill(4) + ".tif" #FOR TEST USE
    image_path = os.path.join(image_folder_path, image_file)
    image_idx = image_file.split("_")[-1].replace(".tif", "")
    print(f"working on image {image_idx}...")
    image = cv2.imread(image_path)

    #ORIGINAL IMAGE
    plt.imshow(image)
    plt.savefig(os.path.join(original_image_dir, image_file.replace(".tif", ".jpg")))

    #PREPROCESSING==================
    image = BackgroundSubtraction(image)#background subtraction
    image = AdjustColorRange(image)#color range adjustment
    plt.imshow(image)
    plt.savefig(os.path.join(preprocessed_image_dir, image_file.replace(".tif", ".jpg")))

    #SEGMENTATION===================
    image, contours = Threshold(image)#define contours
    #save segmented image
    plt.imshow(image)
    plt.savefig(os.path.join(segmented_image_dir, image_file.replace(".tif", ".jpg")))
    plt.close()

    #DATA EXTRACTION=================
    #extract shape metrics of contours on thresholded image
    shape_metrics = DataExtraction(segm_output_csv, image_idx, contours)
    cols = ["image_idx", "spore_id", "x", "y", "perimeter", "area", "minor_axis_length"]
    write_data(shape_metrics, cols, test_idx, segm_output_csv)

  #MAKE GIFS=========
  Gif(original_image_dir, f'{experiment}_{microscopy}_Unprocessed.gif', 3, segmentation_repo)
  Gif(preprocessed_image_dir, f'{experiment}_{microscopy}_Preprocessed.gif', 3, segmentation_repo)
  Gif(segmented_image_dir, f'{experiment}_{microscopy}_Segmented.gif', 3, segmentation_repo)
  

  #PLOTTING FEATURES===============
  for feature in ["perimeter", "area", "minor_axis_length"]:
    LineplotTrackFeature(segm_output_csv, f"{feature}_tmp.jpg", feature)

