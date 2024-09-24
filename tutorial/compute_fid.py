import os
from pytorch_fid import fid_score
from pytorch_fid.inception import InceptionV3
import numpy as np
import torch
import torchvision.transforms as TF
from torch.nn.functional import adaptive_avg_pool2d
from tqdm import tqdm
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path

class ImagePathDataset(torch.utils.data.Dataset):
    def __init__(self, files, transforms=None):
        self.files = files
        self.transforms = transforms

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        img = Image.open(path).convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)
        return img
    
def get_activations(files, model, batch_size=50, dims=2048, device='cpu',
                    num_workers=1, transforms=TF.ToTensor()):
    """Calculates the activations of the pool_3 layer for all images.

    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : Batch size of images for the model to process at once.
                     Make sure that the number of samples is a multiple of
                     the batch size, otherwise some samples are ignored. This
                     behavior is retained to match the original FID score
                     implementation.
    -- dims        : Dimensionality of features returned by Inception
    -- device      : Device to run calculations
    -- num_workers : Number of parallel dataloader workers

    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    model.eval()

    if batch_size > len(files):
        print(('Warning: batch size is bigger than the data size. '
               'Setting batch size to data size'))
        batch_size = len(files)

    dataset = ImagePathDataset(files, transforms=transforms)
    print(f"total: {len(dataset)}")
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             drop_last=False,
                                             num_workers=num_workers)

    pred_arr = np.empty((len(files), dims))

    start_idx = 0

    for batch in tqdm(dataloader):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch)[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred = pred.squeeze(3).squeeze(2).cpu().numpy()

        pred_arr[start_idx:start_idx + pred.shape[0]] = pred

        start_idx = start_idx + pred.shape[0]

    return pred_arr

def calculate_activation_statistics(files, model, batch_size=50, dims=2048,
                                    device='cpu', num_workers=1, transforms=TF.ToTensor()):
    """Calculation of the statistics used by the FID.
    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : The images numpy array is split into batches with
                     batch size batch_size. A reasonable batch size
                     depends on the hardware.
    -- dims        : Dimensionality of features returned by Inception
    -- device      : Device to run calculations
    -- num_workers : Number of parallel dataloader workers

    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the inception model.
    """
    act = get_activations(files, model, batch_size, dims, device, num_workers, transforms)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def compute_fid(files1, files2, model, batch_size, dims, device, num_workers=1,transforms=TF.ToTensor()):
    m1, s1 = calculate_activation_statistics(files1, model, batch_size, dims, device, num_workers, transforms)
    m2, s2 = calculate_activation_statistics(files2, model, batch_size, dims, device, num_workers, transforms)
    fid_value = fid_score.calculate_frechet_distance(m1, s1, m2, s2)
    print("FID: ", fid_value)
    
    
# 准备真实数据分布和生成模型的图像数据
real_images_folder = 'datasets/bolt_compute_fid/good/'
generated_images_folder = 'datasets/bolt_compute_fid/sdxl_bolt_res_512/'

batch_size = 256
device = "cuda"
hidden_dims = 2048
num_workers = 8
block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[hidden_dims]
model = InceptionV3([block_idx]).to(device)
transform = transforms.Compose([
    transforms.Resize(512),
    transforms.CenterCrop(512),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

files1 = list()
files2 = list()

generated_files = Path(generated_images_folder).rglob("./*.png")
for file in generated_files:
    
    fake_path = str(file)
    real_path = fake_path.replace(generated_images_folder, real_images_folder).replace("_.png", ".png")
    
    if os.path.exists(fake_path) and os.path.exists(real_path):
        files1.append(fake_path)
        files2.append(real_path)

# run
compute_fid(files1, files2, model, batch_size, hidden_dims, device, num_workers, transform)
    
