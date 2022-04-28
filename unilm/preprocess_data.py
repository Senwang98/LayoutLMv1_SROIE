import os
import glob
import json
import random
from pathlib import Path
from difflib import SequenceMatcher


import cv2
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from IPython.display import display
import matplotlib
from matplotlib import pyplot, patches

sroie_folder_path = Path("./SROIE2019")
example_file = Path("X51005365187.txt")


def read_bbox_and_words(path: Path):
    '''
        读取path路径下的bbox文件，返回pd形式
    '''
    bbox_and_words_list = []

    with open(path, "r", errors="ignore") as f:
        for line in f.read().splitlines():
            if len(line) == 0:
                continue

            split_lines = line.split(",")

            bbox = np.array(split_lines[0:8], dtype=np.int32)
            text = ",".join(split_lines[8:])

            # From the splited line we save (filename, [bounding box points], text line).
            # The filename will be useful in the future
            bbox_and_words_list.append([path.stem, *bbox, text])

    dataframe = pd.DataFrame(
        bbox_and_words_list,
        columns=["filename", "x0", "y0", "x1", "y1", "x2", "y2", "x3", "y3", "line"],
        # dtype=np.int16,
    ) 
    dataframe = dataframe.drop(columns=["x1", "y1", "x3", "y3"])

    return dataframe


# Example usage
bbox_file_path = sroie_folder_path / "test/box" / example_file
bbox = read_bbox_and_words(path=bbox_file_path)
# print("\n== Dataframe ==")
# print(bbox.head(5))

def read_entities(path: Path):
    with open(path, "r") as f:
        data = json.load(f)

    dataframe = pd.DataFrame([data])
    return dataframe


# Example usage
entities_file_path = sroie_folder_path / "test/entities" / example_file
entities = read_entities(path=entities_file_path)
# print("\n\n== Dataframe ==")
# print(entities)
# exit()

# Assign a label to the line by checking the similarity
# of the line and all the entities
# 针对每一行的ocr内容给其打标签，只有company、data、address、total四类标签，不然就返回other标签
def assign_line_label(line: str, entities: pd.DataFrame):
    line_set = line.replace(",", "").strip().split()
    # print("line_set = ", line_set)
    for i, column in enumerate(entities):
        entity_values = entities.iloc[0, i].replace(",", "").strip()
        entity_set = entity_values.split()
        # print("column = ", column)
        # print("entity_set = ", entity_set)

        matches_count = 0 # 表示文本中有多少被匹配
        for l in line_set:
            if any(SequenceMatcher(a=l, b=b).ratio() > 0.8 for b in entity_set):
                matches_count += 1
            # ===> For debug mode
            # for b in entity_set:
            #     print("Matching ==> ", l, " | ", b, " | ", SequenceMatcher(a=l, b=b).ratio())

            if (
                (column.upper() == "ADDRESS" and (matches_count / len(line_set)) >= 0.5)
                or (column.upper() != "ADDRESS" and (matches_count == len(line_set)))
                or matches_count == len(entity_set)
            ): # 如果是address，只要有一半文本匹配上就算成功，如果没有地址则需要全部匹配上才行；或者当出现实体短于文本的时候这部分文本也算。
                # print("matches_count = ", matches_count)
                return column.upper()

    return "O"


line = bbox.loc[1, "line"] # 取bbox，idx=1的line列，即ocr文本识别内容
label = assign_line_label(line, entities)
# print("Line:", line)
# print("Assigned label:", label)
# exit()

def assign_labels(words: pd.DataFrame, entities: pd.DataFrame):
    max_area = {"TOTAL": (0, -1), "DATE": (0, -1)}  # Value, index
    already_labeled = {
        "TOTAL": False,
        "DATE": False,
        "ADDRESS": False,
        "COMPANY": False,
        "O": False,
    }

    # Go through every line in $words and assign it a label
    labels = []
    for i, line in enumerate(words["line"]):
        label = assign_line_label(line, entities)

        already_labeled[label] = True
        if (label == "ADDRESS" and already_labeled["TOTAL"]) or (
            label == "COMPANY" and (already_labeled["DATE"] or already_labeled["TOTAL"])
        ):
            label = "O"

        # Assign to the largest bounding box
        if label in ["TOTAL", "DATE"]:
            x0_loc = words.columns.get_loc("x0")
            bbox = words.iloc[i, x0_loc : x0_loc + 4].to_list()
            area = (bbox[2] - bbox[0]) + (bbox[3] - bbox[1])

            if max_area[label][0] < area:
                max_area[label] = (area, i)

            label = "O"

        labels.append(label)

    labels[max_area["DATE"][1]] = "DATE"
    labels[max_area["TOTAL"][1]] = "TOTAL"

    words["label"] = labels
    return words


# Example usage
bbox_labeled = assign_labels(bbox, entities)
# print(bbox_labeled.head(15))
# exit()

def split_line(line: pd.Series):
    line_copy = line.copy()

    line_str = line_copy.loc["line"]
    words = line_str.split(" ")

    # Filter unwanted tokens
    words = [word for word in words if len(word) >= 1]

    x0, y0, x2, y2 = line_copy.loc[["x0", "y0", "x2", "y2"]]
    bbox_width = x2 - x0

    new_lines = []
    for index, word in enumerate(words):
        x2 = x0 + int(bbox_width * len(word) / len(line_str))
        line_copy.at["x0", "x2", "line"] = [x0, x2, word]
        new_lines.append(line_copy.to_list())
        x0 = x2 + 5

    return new_lines


# # Example usage
# new_lines = split_line(bbox_labeled.loc[1])
# print("Original row:")
# display(bbox_labeled.loc[1:1, :])

# print("Splitted row:")
# print(pd.DataFrame(new_lines, columns=bbox_labeled.columns))

from time import perf_counter


def dataset_creator(folder: Path):
    bbox_folder = folder / "box"
    entities_folder = folder / "entities"
    img_folder = folder / "img"

    # Sort by filename so that when zipping them together
    # we don't get some other file (just in case)
    entities_files = sorted(entities_folder.glob("*.txt"))
    bbox_files = sorted(bbox_folder.glob("*.txt"))
    img_files = sorted(img_folder.glob("*.jpg"))

    data = []

    print("Reading dataset:")
    for bbox_file, entities_file, img_file in tqdm(
        zip(bbox_files, entities_files, img_files), total=len(bbox_files)
    ):
        # Read the files
        bbox = read_bbox_and_words(bbox_file)
        entities = read_entities(entities_file)
        image = Image.open(img_file)

        # Assign labels to lines in bbox using entities
        bbox_labeled = assign_labels(bbox, entities)
        del bbox

        # Split lines into separate tokens
        new_bbox_l = []
        for index, row in bbox_labeled.iterrows():
            new_bbox_l += split_line(row)
        new_bbox = pd.DataFrame(new_bbox_l, columns=bbox_labeled.columns)
        del bbox_labeled


        # Do another label assignment to keep the labeling more precise
        for index, row in new_bbox.iterrows():
            label = row["label"]

            if label != "O":
                entity_values = entities.iloc[
                    0, entities.columns.get_loc(label.lower())
                ]
                entity_set = entity_values.split()

                # 如果当前文本与label中有一段区间是匹配的，那么认为整个就匹配上了。
                if any(
                    SequenceMatcher(a=row["line"], b=b).ratio() > 0.7
                    for b in entity_set
                ):
                    label = "S-" + label
                else:
                    label = "O"

            new_bbox.at[index, "label"] = label

        width, height = image.size

        data.append([new_bbox, width, height])

    return data


dataset_train = dataset_creator(sroie_folder_path / 'train')
dataset_test = dataset_creator(sroie_folder_path / 'test')

# 根据LayOutLM的论文归一化bbox
def normalize(points: list, width: int, height: int) -> list:
    x0, y0, x2, y2 = [int(p) for p in points]

    x0 = int(1000 * (x0 / width))
    x2 = int(1000 * (x2 / width))
    y0 = int(1000 * (y0 / height))
    y2 = int(1000 * (y2 / height))

    return [x0, y0, x2, y2]


def write_dataset(dataset: list, output_dir: Path, name: str):
    print(f"Writing {name}ing dataset:")
    with open(output_dir / f"{name}.txt", "w+", encoding="utf8") as file, open(
        output_dir / f"{name}_box.txt", "w+", encoding="utf8"
    ) as file_bbox, open(
        output_dir / f"{name}_image.txt", "w+", encoding="utf8"
    ) as file_image:

        # Go through each dataset
        for datas in tqdm(dataset, total=len(dataset)):
            data, width, height = datas

            filename = data.iloc[0, data.columns.get_loc("filename")]

            # Go through every row in dataset
            for index, row in data.iterrows():
                bbox = [int(p) for p in row[["x0", "y0", "x2", "y2"]]]
                normalized_bbox = normalize(bbox, width, height)

                file.write("{}\t{}\n".format(row["line"], row["label"]))
                file_bbox.write(
                    "{}\t{} {} {} {}\n".format(row["line"], *normalized_bbox)
                )
                file_image.write(
                    "{}\t{} {} {} {}\t{} {}\t{}\n".format(
                        row["line"], *bbox, width, height, filename
                    )
                )

            # Write a second newline to separate dataset from others
            file.write("\n")
            file_bbox.write("\n")
            file_image.write("\n")


dataset_directory = Path("./data/working", "dataset")
dataset_directory.mkdir(parents=True, exist_ok=True)

write_dataset(dataset_train, dataset_directory, 'train')
write_dataset(dataset_test, dataset_directory, 'test')

# 创建label文件
# Creating the 'labels.txt' file to the the model what categories to predict.
labels = ["COMPANY", "DATE", "ADDRESS", "TOTAL"]
IOB_tags = ["S"]
with open(dataset_directory / "labels.txt", "w") as f:
    for tag in IOB_tags:
        for label in labels:
            f.write(f"{tag}-{label}\n")
    # Writes in the last label O - meant for all non labeled words
    f.write("O")

# 以下代码可忽略
# pretrained_model_folder_input = sroie_folder_path / Path(
#     "layoutlm-base-uncased"
# )  # Define it so we can copy it into our working directory

# pretrained_model_folder = Path("./kaggle/working/layoutlm-base-uncased/")
# label_file = Path(dataset_directory, "labels.txt")

# # Move to the script directory
# os.chdir("unilm/layoutlm/deprecated/examples/seq_labeling")
