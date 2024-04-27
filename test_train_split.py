# # -*- coding: utf-8 -*-
# """
# Created on Fri Mar  2 15:42:22 2018
#
# @author: yashr
# """
#
# import pandas as pd
# from sklearn.model_selection import train_test_split
#
# labels = pd.read_csv('labels/labels.csv')
#
# train, val = train_test_split(labels, train_size=0.8, random_state=0)
#
# train.to_csv('labels/train.csv')
# val.to_csv('labels/val.csv')


# import os
# import pandas as pd
#
# # Directory containing your test images
# test_images_dir = './test_images'
#
# # List all jpg files in the directory
# image_files = [f for f in os.listdir(test_images_dir) if f.endswith('.jpg')]
#
# # Extract IDs (assuming the ID is the filename without the extension)
# image_ids = [os.path.splitext(f)[0] for f in image_files]
#
# # Create a DataFrame and save as CSV
# df = pd.DataFrame(image_ids, columns=['id'])
# csv_file = './test_images/test_ids.csv'  # Adjust path as needed
# df.to_csv(csv_file, index=False)
#
# print(f"Created CSV file: {csv_file}")


import pandas as pd

# Replace this with the path to your training data CSV file
train_csv_path = './labels/train.csv'

# Load the training data
train_data = pd.read_csv(train_csv_path)

# Get unique breeds
unique_breeds = train_data['breed'].unique()

# Create a DataFrame for class names
class_names_df = pd.DataFrame({
    'index': range(len(unique_breeds)),
    'breed': unique_breeds
})

# Save to CSV
class_names_csv_path = './class_names.csv'  # Adjust this path as needed
class_names_df.to_csv(class_names_csv_path, index=False)

print(f"Class names saved to {class_names_csv_path}")
