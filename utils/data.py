import os, random
from typing import List
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms, datasets
IMAGENET_MEAN=[0.485,0.456,0.406]; IMAGENET_STD=[0.229,0.224,0.225]
def get_transforms(size=224):
    return transforms.Compose([transforms.Resize((size,size)),transforms.ToTensor(),transforms.Normalize(IMAGENET_MEAN,IMAGENET_STD)])
class ImagePathDataset(Dataset):
    def __init__(self, paths: List[str], transform=None):
        self.paths=paths; self.transform=transform or get_transforms()
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        p=self.paths[idx]; img=Image.open(p).convert("RGB"); x=self.transform(img); return x,p
def load_gtsrb_imagefolder(data_dir: str, size=224):
    tfm=get_transforms(size); 
    train_ds=datasets.ImageFolder(os.path.join(data_dir,"train"),transform=tfm)
    test_ds =datasets.ImageFolder(os.path.join(data_dir,"test"), transform=tfm)
    return train_ds, test_ds
def split_qpool_dtest_from_test(test_ds, qpool_n=6000, dtest_n=2000, seed=42):
    rng=random.Random(seed); idxs=list(range(len(test_ds))); rng.shuffle(idxs)
    if qpool_n+dtest_n>len(idxs): raise ValueError("test 샘플 수 부족")
    q_idx=idxs[:qpool_n]; d_idx=idxs[qpool_n:qpool_n+dtest_n]
    q_paths=[test_ds.samples[i][0] for i in q_idx]
    d_paths=[test_ds.samples[i][0] for i in d_idx]
    d_labels=[test_ds.samples[i][1] for i in d_idx]
    return q_paths,(d_paths,d_labels)
