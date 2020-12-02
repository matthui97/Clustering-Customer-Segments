import tensorflow as tf
import argparse
from PIL import Image
import json
from image_preprocessing import process_image
import tensorflow_hub as hub
import numpy as np


def main():
  # Read arguments from command line
    
    def get_input_args():
        parser = argparse.ArgumentParser()
        # Input Command line parameter. Reads input image path.  
        parser.add_argument('input_image', action='store', type=str, 
                             help='input image path')
        parser.add_argument('model_path', action='store', type=str, 
                             help='call model')
        parser.add_argument('--top_k', action='store', type=int, 
                             help='number of top predictions')
        parser.add_argument('--category_names', action='store', type=str, 
                             default='label_map.json')
            
        return parser.parse_args()
        
    in_args = get_input_args()
    
    def load_model(model_path):
        return tf.keras.models.load_model(model_path, custom_objects={'KerasLayer':hub.KerasLayer})

    model = load_model(in_args.model_path)
   
    def load_class_names(label_map):
        with open(label_map, 'r') as f:
            class_names = json.load(f)
            class_names_dict = dict()
            for key in class_names:
                class_names_dict[str(int(key))] = class_names[key]
            return class_names_dict
    
    class_names = load_class_names(in_args.category_names)
    
        
    def predict(image_path, model, top_k):
        im = Image.open(image_path)
        image = np.asarray(im)
        processed_image = process_image(image)
        expanded_dim_image = np.expand_dims(processed_image,axis = 0)
        ps = model.predict(expanded_dim_image)
        probs, classes = tf.math.top_k(ps, k=top_k, sorted=False)
        for idx, clazz in enumerate(classes.numpy()[0]):
            flower_name = class_names[str(clazz+1)]
            return flower_name
            return probs[idx]
        
    
    print(predict(in_args.input_image, model, in_args.top_k))
    
    
    

if __name__ == "__main__":
    main()
