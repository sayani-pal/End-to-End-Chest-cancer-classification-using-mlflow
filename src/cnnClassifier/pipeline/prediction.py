import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os


class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename

    def predict(self):

        # Load trained model
        model_path = os.path.join(
            "artifacts",
            "training",
            "model.h5"
        )

        model = load_model(model_path)

        # Load and preprocess image
        imagename = self.filename

        test_image = image.load_img(
            imagename,
            target_size=(224, 224)
        )

        # Convert image to array
        test_image = image.img_to_array(test_image)

        # Normalize image
        test_image = test_image / 255.0

        # Add batch dimension
        test_image = np.expand_dims(test_image, axis=0)

        # Predict
        result = model.predict(test_image)

        # Debug outputs
        print("Raw Prediction:", result)

        # Get predicted class index
        predicted_class = np.argmax(result, axis=1)[0]

        print("Predicted Class Index:", predicted_class)

        # Class labels
        class_labels = {
            0: "Adenocarcinoma Cancer",
            1: "Normal"
        }

        prediction = class_labels[predicted_class]

        print("Final Prediction:", prediction)

        return [{"image": prediction}]