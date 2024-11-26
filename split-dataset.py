import os
import shutil
from tqdm import tqdm

def split_dataset():
    root_path = r"./data"
    input_path = r"leftImg8bit_trainvaltest_foggy/leftImg8bit_foggy"
    target_path = r"gtFine_trainvaltest/gtFine"
    input_beta = 0.01

    city_to_split = {
        'train': ['aachen', 'bochum', 'bremen', 'cologne', 'darmstadt', 'dusseldorf', 'erfurt', 'hamburg', 'hanover', 'jena', 'monchengladbach', 'strasbourg', 'stuttgart', 'tubingen', 'ulm', 'zurich'],
        'val': ['frankfurt', 'krefeld'],
        'test': ['lindau', 'munster', 'weimar']
    }

    for split, cities in city_to_split.items():
        output_images_path = os.path.join(root_path, '..', 'data_split', 'leftImg8bit_trainvaltest_foggy', 'leftImg8bit_foggy', split)
        output_annotations_path = os.path.join(root_path, '..', 'data_split', 'gtFine_trainvaltest', 'gtFine', split)

        if not os.path.exists(output_images_path):
            os.makedirs(output_images_path, exist_ok=True)
        if not os.path.exists(output_annotations_path):
            os.makedirs(output_annotations_path, exist_ok=True)

        for city_name in tqdm(cities, desc=f"Processing {split} cities"):
            city_input_path = find_city_path(root_path, input_path, city_name)
            city_target_path = find_city_path(root_path, target_path, city_name)
            city_output_images_path = os.path.join(output_images_path, city_name)
            city_output_annotations_path = os.path.join(output_annotations_path, city_name)

            if not os.path.exists(city_output_images_path):
                os.makedirs(city_output_images_path)
            if not os.path.exists(city_output_annotations_path):
                os.makedirs(city_output_annotations_path)

            for file_name in os.listdir(city_input_path):
                if str(input_beta) in file_name:
                    input_file_path = os.path.join(city_input_path, file_name)
                    target_file_name = file_name.replace('leftImg8bit', 'gtFine_labelIds')
                    target_file_path = os.path.join(city_target_path, target_file_name)

                    if os.path.exists(target_file_path):
                        shutil.copy(input_file_path, city_output_images_path)
                        shutil.copy(target_file_path, city_output_annotations_path)

            # Copy all contents of the city_input_path to city_output_images_path
            for item in os.listdir(city_input_path):
                s = os.path.join(city_input_path, item)
                d = os.path.join(city_output_images_path, item)
                if os.path.isdir(s):
                    shutil.copytree(s, d, dirs_exist_ok=True)
                else:
                    shutil.copy2(s, d)

            # Copy all contents of the city_target_path to city_output_annotations_path
            for item in os.listdir(city_target_path):
                s = os.path.join(city_target_path, item)
                d = os.path.join(city_output_annotations_path, item)
                if os.path.isdir(s):
                    shutil.copytree(s, d, dirs_exist_ok=True)
                else:
                    shutil.copy2(s, d)

def find_city_path(root_path, base_path, city_name):
    for split in ['train', 'val', 'test']:
        potential_path = os.path.join(root_path, base_path, split, city_name)
        if os.path.exists(potential_path):
            return potential_path
    raise FileNotFoundError(f"City {city_name} not found in any split.")

if __name__ == "__main__":
    split_dataset()