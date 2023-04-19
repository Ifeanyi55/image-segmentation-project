# load modules
from metaseg import SegAutoMaskPredictor
from IPython.display import Image

# define predictor function
def AutoSegmentImage(image):
  autoseg_image = SegAutoMaskPredictor().image_predict(
    source = image,
    model_type = "vit_l",
    points_per_side=16,
    points_per_batch=64,
    output_path="output.jpg",
    min_area=0,
    save = True
    )
  
  return Image("output.jpg")


# segment image
AutoSegmentImage(image = "/content/sample_data/parasite2.jpg")