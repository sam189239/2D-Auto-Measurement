import sys
sys.path.append("hmr2\\src")
from utils import *

if __name__ == "__main__":
  ## Loading Models ##
  start_load = time.time()
  seg_model = segmentation_model()
  hmr_model = HMRmodel()
  ## Renderer ## - optional
  # from visualise.trimesh_renderer import TrimeshRenderer
  # renderer = TrimeshRenderer()
  end_load = time.time()
  print("Loaded Models... Time taken: " + str(end_load-start_load)+ " seconds.")
  while True:
    ht = input("Enter height in cm: ")
    if ht == "0":
      break
    process(float(ht))
