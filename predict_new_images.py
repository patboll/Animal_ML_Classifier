import numpy as np
import tensorflow as tf
from keras import preprocessing
from PIL import Image
import matplotlib.pyplot as plt
import keras.preprocessing.image as image
from keras.applications.mobilenet_v2 import preprocess_input

# from animal_species_classifier import my_var

model = tf.keras.models.load_model("animal_classifier_4.h5")
# model = tf.keras.models.load_model("C:\Users\Patrick\AppData\Local\Programs\Microsoft VS Code\animal_classifier.h5")

# print(my_var+1)

with open('C:\\Users\\Patrick\\Github\\archive\\name_of_the_animals.txt', 'r') as file:
    content = file.read()
    class_names = content.splitlines()
#     # print(lines)

def predict_image(model, img_path, class_names):

    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # img_array = preprocess_input(np.array(img))
    # img_array = np.expand_dims(img_array, axis=0)  

    predictions = model.predict(img_array)[0]
    
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions)

    print("Prediction:", predicted_class)
    print("Confidence:", confidence)

    top = np.argsort(predictions)[-5:][::-1]
    for i in top:
        print(class_names[i], predictions[i])

    # Show image
    plt.imshow(img)
    plt.title(f"{predicted_class} ({confidence:.2f})")
    plt.axis("off")
    plt.show()
    
# Example
predict_image(model, r"C:\Users\Patrick\Github\archive\animals\test_3.jpg", class_names)

# print(model.summary())
# print(model.output_shape)