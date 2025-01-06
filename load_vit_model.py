import os
import pandas as pd
import numpy as np
import csv
import glob
import argparse
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from timm.models.vision_transformer import VisionTransformer


parser = argparse.ArgumentParser(description='Getting features from Dino.')
parser.add_argument('--phase', type=str, default='train', help='name.')
parser.add_argument('--c', type=str, default='adc', help='name of the file that contains image paths.')
parser.add_argument('--group', type=str, default='05', help='name.')
args = parser.parse_args()

print('args: ', args)

def get_pretrained_url(key):
    URL_PREFIX = "https://github.com/lunit-io/benchmark-ssl-pathology/releases/download/pretrained-weights"
    model_zoo_registry = {
        "DINO_p16": "dino_vit_small_patch16_ep200.torch",
        "DINO_p8": "dino_vit_small_patch8_ep200.torch",
    }
    pretrained_url = f"{URL_PREFIX}/{model_zoo_registry.get(key)}"
    return pretrained_url


def vit_small(pretrained, progress, key, **kwargs):
    patch_size = kwargs.get("patch_size", 16)
    model = VisionTransformer(
        img_size=224, patch_size=patch_size, embed_dim=384, num_heads=6, num_classes=0)
    #ToDo: check if it accepts img_size=256

    if pretrained:
        pretrained_url = get_pretrained_url(key)
        verbose = model.load_state_dict(
            torch.hub.load_state_dict_from_url(pretrained_url, progress=progress)
        )
        print(verbose)
    return model

class roi_dataset(Dataset):
    def __init__(self, img_csv,
                 ):
        super().__init__()
        self.transform = transforms.Compose(
                        [
                        transforms.Resize(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))
                        ]
        )

        self.images_lst = img_csv

    def __len__(self):
        return len(self.images_lst)

    def __getitem__(self, idx):
        path = self.images_lst.filename[idx]
        image = Image.open(path).convert('RGB')
        image = self.transform(image)


        return image

def create_csv(case):
    case_list = case.split('/')
    case_name = case_list[-1]
    csv_filename = '{}_{}_{}.csv'.format(args.phase, args.c, case_name)
    if(not os.path.exists(csv_filename)):
        with open(csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['filename'])  # Header row
            filenames = glob.glob(os.path.join(case, '*'))
            print('number of images found = ', len(filenames))
            for path in filenames:
                writer.writerow([path])
    print('done creating {}_{}_{}.csv'.format(args.phase, args.c, case_name))
    img_csv = pd.read_csv('{}_{}_{}.csv'.format(args.phase, args.c, case_name))
    return img_csv, case_name

if __name__ == "__main__":
    # initialize ViT-S/16 trunk using DINO pre-trained weight
    model = vit_small(pretrained=True, progress=False, key="DINO_p16", patch_size=16)
    print('running eval')
    model.eval()

    base_path = '/common/deogun/alali/data/lung_png20x/'
    out_path = 'features/'
    if(not os.path.exists(out_path)):
        os.mkdir(out_path)
    out_phase_path = os.path.join(out_path, args.phase)
    if(not os.path.exists(out_phase_path)):
        os.mkdir(out_phase_path)
    out_class_path = os.path.join(out_phase_path, args.c)
    if(not os.path.exists(out_class_path)):
        os.mkdir(out_class_path)
    print('output path: ', out_class_path)
    phase_path = os.path.join(base_path, args.phase)
    class_path = os.path.join(phase_path, args.c)
    case_path = glob.glob(os.path.join(class_path, 'TCGA-{}*'.format(args.group)))
    for k, case in enumerate(case_path):
        print('case [{}/{}]= {}'.format(k, len(case_path), case))
        img_csv, case_name = create_csv(case)
        test_datat=roi_dataset(img_csv)
        database_loader = DataLoader(test_datat, batch_size=1, shuffle=False)
        feat_list = []
        #sample_size = 0
        with torch.no_grad():
            for i, batch in enumerate(database_loader):
                features = model(batch)
                features = features.cpu().numpy()
                features = features.squeeze()
                if(i % 100 == 0):
                    print('== loading batch [{}/{}] batch of shape: {}'.format(i, len(database_loader), batch.shape))
                    print('=== extracted features of shape = ', features.shape)
                    #print('=== size of one feature vector in bytes = ', sample_size)
            feat_list.append(features)
        print('done array of length = ', len(feat_list))
        print('Trying to convert the list to numpy array...')
        feat_np = np.array(feat_list)
        print('done np array of shape: ', feat_np.shape)
        save_path = os.path.join(out_class_path, case_name)
        print('trying to save array in path: ', save_path)
        np.save(arr=feat_np, file=save_path)
        print('np array succ saved')
    print('============== done =====================')


