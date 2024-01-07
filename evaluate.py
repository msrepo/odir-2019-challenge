import os
from models.models import Net
from transforms.transforms import LABELS, get_img_transform
import torch
import argparse
from os.path import join

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", "-c", required=True)
parser.add_argument("--device", "-d", required=False)

args = parser.parse_args()


if args.device:
    device = torch.device(args.device)
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TEST_DATA_ROOT_DIR = "odir2019/ODIR-5K_Testing_Images"
TEST_CSV = "csv/processed_test_ODIR-5k.csv"

IMG_SIZE = 300
N_CHANNELS = 3
N_CLASSES = 8
BATCH_SIZE = 32


model = Net().to(device)
# load checkpoint
model.load_state_dict(torch.load(args.checkpoint))
model.eval()

data_transform = get_img_transform(img_size=IMG_SIZE)


def get_img_paths(data_root_dir, suffix=".jpg"):
    from glob import glob
    from os.path import join

    img_paths = glob(join(f"{data_root_dir}/*{suffix}"))
    return img_paths


def predict(img_path, model,device=device):
    img = data_transform(img_path)
    # add batch dimension
    pred_proba = model(img.unsqueeze(0).to(device))
    return pred_proba

def get_most_probable(pred_proba):
    return LABELS[torch.argmax(pred_proba)], torch.max(pred_proba) 

def save_predictions(ids, predictions, save_csv_path):
    import pandas as pd

    class_labels = ['N','D','G','C','A','H','M','O']
    row_header = ['ID',]+class_labels
    
    df = pd.DataFrame(columns=row_header)
    for i,(id,p) in enumerate(zip(ids, predictions)):
        one_hot_prediction = [0,]*len(class_labels)
        idx = class_labels.index(p)
        one_hot_prediction[idx] = 1
        df.loc[i] = [id,]+ one_hot_prediction
    df.to_csv(index=False, path_or_buf=save_csv_path)


def get_ID_from_path(path: str):
    # odir2019/ODIR-5K_Testing_Images/937_left.jpg -> 937
    from pathlib import Path

    sub_parts = Path(path).stem.split("_")

    assert len(sub_parts) == 2

    return sub_parts

def merge_predictions(left_pred_proba, right_pred_proba):
    if left_pred_proba is None:
        return get_most_probable(right_pred_proba)[0]
    if right_pred_proba is None:
        return get_most_probable(left_pred_proba)[0]
    
    left_label, left_proba = get_most_probable(left_pred_proba)
    right_label, right_proba = get_most_probable(right_pred_proba)

    if left_pred_proba == 'N' and right_pred_proba == 'N':
        return 'N'
    else:
        if left_proba >= right_proba:
            return left_label
        else:
            return right_label

left_img_paths = sorted(get_img_paths(TEST_DATA_ROOT_DIR, suffix="_left.jpg"))
right_img_paths = sorted(get_img_paths(TEST_DATA_ROOT_DIR, suffix="_right.jpg"))

left_ids = [ get_ID_from_path(p)[0] for p in left_img_paths]
right_ids = [get_ID_from_path(p)[0] for p in right_img_paths]

ids = set(left_ids + right_ids)

N_SAMPLES = len(list(ids))
predictions = []
for id in list(ids)[:N_SAMPLES]:
    left_image = join(TEST_DATA_ROOT_DIR,f'{id}_left.jpg')
    if os.path.exists(left_image):
        left_prediction = predict(left_image,model)
    else:
        left_prediction = None

    right_image = join(TEST_DATA_ROOT_DIR,f'{id}_right.jpg')
    if os.path.exists(right_image):
        right_prediction =  predict(right_image,model)
    else:
        right_prediction = None

    merge_p = merge_predictions(left_prediction,right_prediction)
    predictions.append(merge_p)

save_predictions(ids=ids, predictions=predictions, save_csv_path='XYZ_ODIR_prediction.csv')
