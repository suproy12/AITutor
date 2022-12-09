
from deepface import DeepFace
import numpy as np
import pandas as pd
import os
import re
#model = DeepFace.build_model("Facenet") 
images_path = 'C:/Users/Supra/Documents/Supravat/Documents/ISB/Capstone Project/FaceImages/SupRay/'

#images_path = os.listdir('food101/images/')
embedding_df = pd.DataFrame()
embeddings= []

dict_obj = {}
for curr_img in os.listdir(images_path):
     embedding = DeepFace.represent(os.path.join(images_path,curr_img),  model_name = "VGG-Face")
     dict_obj[curr_img] = embedding
 
os.getcwd()
import pickle
# save dictionary to pickle file
with open('image_verification.pickle', 'wb') as file:
    pickle.dump(dict_obj, file, protocol=pickle.HIGHEST_PROTOCOL)
with open("image_verification.pickle", "rb") as file:
    loaded_dict = pickle.load(file)
    





def findCosineDistance(source_representation, test_representation):
    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))





def findSimilarImage(face_embedding) :
    for key in loaded_dict :
        result = findCosineDistance(face_embedding , loaded_dict[key] )
    #print(key, result)
        if result < 0.4 :
            print('The captured face is matching with :' , key)
            captured_image = key
            break
        else :
           captured_image = 'Not matching'
        print(captured_image)
    print(captured_image)
    return(re.sub("[^a-z,^A-Z]", "",captured_image))






