#### This script should process and image based on an exitsing trained model
# Argument 1 should be the number correspoding to the args[x].json the model is based on

import sys

from helpers import *
from dataset import MRIDataset
import torch
from UNet import UNetModel
from GaussianDiffusion import GaussianDiffusionModel, get_beta_schedule
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import nibabel as nib
from torchvision import datasets, transforms

ROOT_DIR = "./"

def cycle(iterable):
    while True:
        for x in iterable:
            yield x

def init_dataset(ROOT_DIR, args):
    """
        Initializes the dataset to analyze
    """
    analyze_dataset = MRIDataset(
            ROOT_DIR=f'{ROOT_DIR}DATASETS/Analyze/', img_size=args['img_size'], random_slice=args['random_slice']
            )
    return analyze_dataset

def init_dataset_loader(mri_dataset, args, shuffle=True):
    """
        Makes a dataset loader to progressively feed the image
    """
    dataset_loader = cycle(
            torch.utils.data.DataLoader(
                    mri_dataset,                    # 23/01/2025 get that dataset which implements the __get_item__ method and __length__
                    batch_size=args['Batch_Size'], shuffle=shuffle,
                    num_workers=0, drop_last=True   # 23/01/2025 drop the last batch if too small (not the case with batch size of 1)
                    )
            )

    return dataset_loader

def check_args():
    """
        checks the system args 
    """
    # Ensure we have exactly two arguments (excluding the script name)
    if len(sys.argv) != 4:
        print("Usage: python process_path.py <args_nr:number> <filename:string> <slice:number/string>")
        sys.exit(1)
    
    # Retrieve arguments
    input_path = sys.argv[1]
    try:
        arg_number = int(sys.argv[1])
    except ValueError:
        print("Error: The first argument must be a number.")
        sys.exit(1)

    try: 
        image_filename = str(sys.argv[2])
    except ValueError:
        print("Error: The second argument must be the filename represented as a string")
        sys.exit(1)
    
    if sys.argv[3] == "all":
        slice_number = "all"
    else:
        try:
            slice_number = int(sys.argv[3])
        except ValueError:
            print("Error the second argument must either be \"all\" or number")
            sys.exit()

    # Construct JSON filename
    json_filename = f"args{arg_number}.json"

    # Check if the JSON file exists
    if not os.path.exists("test_args/" + json_filename):
        print(f"Error: JSON file 'test_args/{json_filename}' does not exist.")
        sys.exit(1)

    return json_filename, image_filename, slice_number

def create_image(x, diffusion, model, ema, args):
    print("computing image prediction")

    row_size = min(8, args['Batch_Size'])
    # for a given t, output x_0, & prediction of x_(t-1), and x_0
    # 24/01/2025 RM get tensor with noise with same dimensions as x (current tensor of input image)
    noise = torch.rand_like(x)
    # 24/01/2025  RM get tensore with noise on random timestepts with same dimensions as x 
    t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=x.device)
    # 24/01/2025 RM get tensore at timepoint t (applied noise)
    x_t = diffusion.sample_q(x, t, noise)
    # 24/01/2025 RM get reconstructed image from the image at current timepoint t
    temp = diffusion.sample_p(ema, x_t, t)
    # 24/01/2025 RM combines the tensores of the input, current noise prediction and current reconstruction
    out = torch.cat(
            (x[:row_size, ...].cpu(), 
            temp["sample"][:row_size, ...].cpu(),
            temp["pred_x_0"][:row_size, ...].cpu())
            )
    plt.title(f'1. real, 2. sample, 3. prediction, x_0-final-epoch')

    plt.rcParams['figure.dpi'] = 150
    plt.grid(False)
    # 24/01/2025 RM make an image / figure out of the current state
    plt.imshow(gridify_output(out, row_size), cmap='gray')
    # 24/01/2025 RM save the figure
    plt.savefig(f'./diffusion-analyzing-images/ARGS={args["arg_num"]}/EPOCH=final-prediction=.png')
    plt.clf()

    print("computing Means squared error image")
    loss, estimates = diffusion.p_loss(model, x, args)
    noisy, est = estimates[1], estimates[2]

    # for a given t, output x_0, x_t, & prediction of noise in x_t & MSE
    out = torch.cat(
            (x[:row_size, ...].cpu(), 
                noisy[:row_size, ...].cpu(), 
                est[:row_size, ...].cpu(),
                (est - noisy).square().cpu()[:row_size, ...])
            )
    # 24/01/2025 RM changed the figure to show what it really means
    plt.title(f'1. real, 2. noisy, 3. noise prediction, 4. difference betwen noise and noise prediction (mse), mse-final-epoch')
    plt.rcParams['figure.dpi'] = 150
    plt.grid(False)
    # 24/01/2025 RM make an image / figure out of the current state
    plt.imshow(gridify_output(out, row_size), cmap='gray')
    # 24/01/2025 RM save the figure
    plt.savefig(f'./diffusion-analyzing-images/ARGS={args["arg_num"]}/EPOCH=final_MSE.png')
    plt.clf()
    
def create_final_image(x, diffusion, model, ema, args):

    print("compute final image")
    row_size = min(8, args['Batch_Size'])
    # Add noise to the input x_0 at fixed timestep λ
    if args["r_lambda"]:
        lambda_timestep = args["r_lambda"]
    else:
        # default 250
        lambda_timestep = 250

    # get noisy image at timepoint lambda
    x_lambda = diffusion.sample_q(
        x, 
        torch.full((x.shape[0],), lambda_timestep, device=x.device), 
        torch.rand_like(x)
    )

    # Reconstruct from the most noisy version
    # Denoise x_lambda back to x_0
    reconstruction = diffusion.sample_p(
        ema, 
        x_lambda, 
        torch.full((x.shape[0],), lambda_timestep, device=x.device)
    )

    # Compute the difference between the original and the reconstructed image
    difference = (x - reconstruction["sample"]).abs()

    # 24/01/2025 RM combines the tensores of the input, current noise prediction and current reconstruction
    out = torch.cat(
        (
            x[:row_size, ...].cpu(), 
            x_lambda[:row_size, ...].cpu(),
            reconstruction["sample"][:row_size, ...].cpu(),
            difference[:row_size, ...].cpu() 
        ),
        dim=0  # Ensure you're specifying the axis (default is 0)
    )

    plt.title(f'1. Input, 2. Max Noise, \n 3. Reconstruction, 4. Difference between Input and Reconstruction \n model: Final Model')

    plt.rcParams['figure.dpi'] = 150
    plt.grid(False)
    # 24/01/2025 RM make an image / figure out of the current state
    plt.imshow(gridify_output(out, row_size), cmap='gray')
    # 24/01/2025 RM save the figure
    plt.savefig(f'./diffusion-analyzing-images/ARGS={args["arg_num"]}/EPOCH=final-reconstruction=.png')
    plt.clf()


def make_prediction(real, recon, x_t, threshold=0.5, error_fn="sq"):
    """
    Make generic prediction and output tensor with order (real, x_lambda, reconstruction, square error, square error
    threshold, ground truth mask)
    :param real: initial real image x_0
    :param recon: reconstruction when diffused to x_t
    :param x_t: middle image when initial image x_0 is noised through t time steps
    :param threshold: value to take threshold
    :param error_fn: square or l1 error - future work could explore error functions in feature space
    :return:
    """
    if error_fn == "sq":
        mse = ((recon - real).square() * 2) - 1
    elif error_fn == "l1":
        mse = (recon - real)
    mse_threshold = mse > (threshold * 2) - 1
    mse_threshold = (mse_threshold.float() * 2) - 1

    return torch.cat((real, x_t, recon, mse, mse_threshold)), mse_threshold

def create_figure(img, diff, unet, args, filename, slicenumber):
    for i in [f'./diffusion-analyzing-images/', f'./diffusion-analyzing-images/ARGS={args["arg_num"]}']:
        if not os.path.exists(i):
            os.makedirs(i)
    print("compute final figure")
    # slice = np.random.choice([0, 1, 2, 3], p=[0.2, 0.3, 0.3, 0.2])
    output_250 = diff.forward_backward(
                unet, img,
                see_whole_sequence="whole",
                # t_distance=5, denoise_fn=args["noise_fn"]
                t_distance=250, denoise_fn=args["noise_fn"]
                )

    output_250_images, mse_threshold_250 = make_prediction(
                img, 
                output_250[-1].to(device),
                output_250[251 // 2].to(device)
                )
    temp = os.listdir(f"./diffusion-analyzing-images/ARGS={args['arg_num']}")

    fig, subplots = plt.subplots(
                1, 5, sharex=True, sharey=True, constrained_layout=False, figsize=(6, 3),
                squeeze=False,
                gridspec_kw={'wspace': 0, 'hspace': 0}
                )
    tempplot = fig.add_subplot(111, frameon=False)

    # Add a title to the figure
    fig.suptitle(f"Diffusion Model Analysis {filename} slice: {slicenumber} ", fontsize=10)

    subplots[0][0].imshow(img.reshape(*args["img_size"]).cpu().numpy(), cmap="gray")
    subplots[0][1].imshow(output_250[251 // 2].reshape(*args["img_size"]).cpu().numpy(), cmap="gray")
    subplots[0][2].imshow(output_250[-1].reshape(*args["img_size"]).cpu().numpy(), cmap="gray")
    subplots[0][3].imshow(output_250_images[3].reshape(*args["img_size"]).cpu().numpy(), cmap="hot")
    subplots[0][4].imshow(output_250_images[4].reshape(*args["img_size"]).cpu().numpy(), cmap="gray")

    for i, val in enumerate(["$x_0$", "$x_t$", "Reconstruction", "Square Error", "Anomaly Prediction"]):
            subplots[0][i].set_xlabel(f"{val}", fontsize=6)
            subplots[0][i].xaxis.set_label_position("top")
    
    subplots[0][0].set_ylabel(f"x_{250}", fontsize=6)
    subplots[0][0].yaxis.set_label_position("left")

    plt.tick_params(axis='both', labelcolor='none', which='both', top=False, left=False, bottom=False, right=False, labelbottom=False, labelleft=False)

    plt.savefig(
                f'./diffusion-analyzing-images/ARGS={args["arg_num"]}/{args["arg_num"]}-AnoDDPM-{filename}-{slicenumber}'
                f'={len(temp) + 1}.png'
                )

    plt.close('all')
    

def get_image(filename):
    img_name = os.path.join(
            f'{ROOT_DIR}DATASETS/Analyze_ABMRI/', filename, f"{filename}_2000002_1.nii.gz"
    )
    # random between 40 and 130
    # print(nib.load(img_name).slicer[:,90:91,:].dataobj.shape)
    # 23/01/2025 RM loading of the new image 
    img = nib.load(img_name)
    image = img.get_fdata()

    # 23/01/2025 RM compute mean, standard deviation and range
    image_mean = np.mean(image)
    
    image_std = np.std(image)
    img_range = (image_mean - 1 * image_std, image_mean + 2 * image_std)
    # 23/01/2025 RM normalize image between 0 and 1
    image = np.clip(image, img_range[0], img_range[1])
    image = image / (img_range[1] - img_range[0])

    return image

def transform(img_size = [256, 256], custom_transform=None):
    """
    Returns a composed transformation pipeline for image preprocessing.
    
    Parameters:
    - img_size (int or tuple): The target size for resizing the image.
    - custom_transform (torchvision.transforms.Compose, optional): A custom transformation pipeline to use instead.
      If provided, this function will return the custom transformation instead.
    
    Returns:
    - torchvision.transforms.Compose: The transformation pipeline.
    """
    if custom_transform:
        return custom_transform
    
    return transforms.Compose([
        transforms.ToPILImage(),  # Convert to PIL Image
        # transforms.RandomAffine(3, translate=(0.02, 0.09)),  # Random affine transformation Not needed for analyzing
        transforms.CenterCrop(256),  # Center crop (may need adaptation)
        transforms.Resize(img_size, transforms.InterpolationMode.BILINEAR),  # Resize to target size
        transforms.ToTensor(),  # Convert back to tensor
        transforms.Normalize((0.5,), (0.5,))  # Normalize values
    ])


def main():
    """
        Load arguments, run training and testing functions, then remove checkpoint directory
    :return:
    """

    args_file, image_filename, slice_number = check_args()
    # Load paramters 
    args, output = load_parameters(device)

    try:
        os.makedirs(f'./diffusion-analyzing-videos/ARGS={args["arg_num"]}')
        os.makedirs(f'./diffusion-analyzing-images/ARGS={args["arg_num"]}')
    except OSError:
        pass

    # get UnetModel
    in_channels =  1
    unet = UNetModel(
            args['img_size'][0], args['base_channels'], channel_mults=args['channel_mults'], in_channels=in_channels
            )
    ema = UNetModel(
            args['img_size'][0], args['base_channels'], channel_mults=args['channel_mults'], in_channels=in_channels
            )

    betas = get_beta_schedule(args['T'], args['beta_schedule'])

    # load diffusion Model
    diffusion = GaussianDiffusionModel(
            args['img_size'], betas, loss_weight=args['loss_weight'],
            loss_type=args['loss-type'], noise=args["noise_fn"]
            )
    
    ema.load_state_dict(output["ema"])
    ema.to(device)
    ema.eval()

    unet.load_state_dict(output["model_state_dict"])
    unet.to(device)
    unet.eval()

    if slice_number == "all":
        image = get_image(image_filename)
        
        for slice_idx in range(80):
            image = image[:, :, slice_idx:slice_idx+1].astype(np.float32)
            
            x = transform([256, 256])(image)
            x = x.unsqueeze(0)
            x = x.to(device)

            create_image(x, diffusion, unet, ema, args)

            create_final_image(x, diffusion, unet, ema, args)

            img = x.reshape(x.shape[1], 1, *args["img_size"])
            create_figure(img, diffusion, unet, args, image_filename, slice_idx)
    else:
        # get the image
        image = get_image(image_filename)
        # get the correct slice
        image = image[:, :, slice_number:slice_number+1].astype(np.float32)
        # transform image
        x = transform([256, 256])(image)  # Get the transform function
        # Ensure correct shape (B, C, H, W)
        x = x.unsqueeze(0)
        # move to device
        x = x.to(device)

        create_image(x, diffusion, unet, ema, args)

        create_final_image(x, diffusion, unet, ema, args)

        img = x.reshape(x.shape[1], 1, *args["img_size"])
        create_figure(img, diffusion, unet, args, image_filename, slice_number)
    

if __name__ == '__main__':
    # get the device 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main()


