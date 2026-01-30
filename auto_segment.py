from autodistill_grounded_sam import GroundedSAM
from autodistill.detection import CaptionOntology
from autodistill.utils import plot
import cv2

# define an ontology to map class names to our GroundedSAM prompt
# the ontology dictionary has the format {caption: class}
# where caption is the prompt sent to the base model, and class is the label that will
# be saved for that caption in the generated annotations
# then, load the model
base_model = GroundedSAM(
    ontology=CaptionOntology(
        {
            "white stripe": "stripe"
        }
    )
)

# run inference on a single image
results = base_model.predict("C:\\azvision\\training\\train\\images\\batch-001-img-2024-08-24T08-26-03.jpg")

plot(
    image=cv2.imread("logistics.jpeg"),
    classes=base_model.ontology.classes(),
    detections=results
)
# label all images in a folder called `context_images`
base_model.label("./context_images", extension=".jpeg")
