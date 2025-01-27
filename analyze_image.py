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

ROOT_DIR = "./"

def cycle(iterable):
    while True:
        for x in iterable:
            yield x

def init_dataset(ROOT_DIR, args):
    """
        Initializes the dataset to analyze
    """
    # TODO adapt completely to our usecase
    analyze_dataset = MRIDataset(
            ROOT_DIR=f'{ROOT_DIR}DATASETS/Analyze/', img_size=args['img_size'], random_slice=args['random_slice']
            )
    return analyze_dataset

def init_dataset_loader(mri_dataset, args, shuffle=True):
    """
        Makes a dataset loader to progressively feed the image
    """
    # TODO really necessairy?
    dataset_loader = cycle(
            # 23/01/2025 makees torch dataloader 
            torch.utils.data.DataLoader(
                    mri_dataset,                # 23/01/2025 get that dataset which implements the __get_item__ method and __length__
                    batch_size=args['Batch_Size'], shuffle=shuffle,
                    num_workers=0, drop_last=True # 23/01/2025 drop the last batch if too small (not the case with batch size of 1)
                    )
            )

    return dataset_loader

def check_args():
    """
        checks the system args 
    """
    # Ensure we have exactly two arguments (excluding the script name)
    if len(sys.argv) != 2:
        print("Usage: python process_path.py <number>")
        sys.exit(1)
    
    # Retrieve arguments
    input_path = sys.argv[1]
    try:
        arg_number = int(sys.argv[1])
    except ValueError:
        print("Error: The first argument must be a number.")
        sys.exit(1)

    # Construct JSON filename
    json_filename = f"args{arg_number}.json"

    # Check if the JSON file exists
    if not os.path.exists("test_args/" + json_filename):
        print(f"Error: JSON file 'test_args/{json_filename}' does not exist.")
        sys.exit(1)

    return json_filename

def create_video(analyzing_dataset_loader, diffusion, ema, args):
    # TODO get running
    plt.rcParams['figure.dpi'] = 200
    if args["save_vids"]: # 22/01/2025 RM TODO remove eventually 
        for i in [*range(100, args['sample_distance'], 100)]:
            data = next(analyzing_dataset_loader)

            x = data["image"]
            x = x.to(device)

            row_size = min(5, args['Batch_Size'])
            print("creating animations")
            fig, ax = plt.subplots()
            out = diffusion.forward_backward(ema, x, see_whole_sequence="half", t_distance=i)
            imgs = [[ax.imshow(gridify_output(x, row_size), animated=True)] for x in out]
            ani = animation.ArtistAnimation(
                    fig, imgs, interval=200, blit=True,
                    repeat_delay=1000
                    )
            print("saving animations")
            files = os.listdir(f'./diffusion-analyzing-videos/ARGS={args["arg_num"]}/test-set/')
            ani.save(f'./diffusion-analyzing-videos/ARGS={args["arg_num"]}/test-set/t={i}-attempts={len(files) + 1}.mp4')

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


def compute_vlb(x, diffusion, model, args):
    """
        :param analyzing_dataset_loader: cycle(dataloader) instance for evaluation
        :param diffusion: GaussianDiffusionModel instance 
        :param model: original unet for VLB calc
        :param args: arguments dictionary form test_args
    """
    print("computing vlb")

    # calulate Variational Lower Bound to have a performance indicator of the difference between input and output
    vlb_terms = diffusion.calc_total_vlb(x, model, args)

    return vlb_terms

def PSNR(recon, real):
    se = (real - recon).square()
    mse = torch.mean(se, dim=list(range(len(real.shape))))
    psnr = 20 * torch.log10(torch.max(real) / torch.sqrt(mse))
    return psnr.detach().cpu().numpy()

def compute_psnr(x, diffusion, ema, args):
    """
        :param analyzing_dataset_loader: cycle(dataloader) instance for evaluation
        :param diffusion: GaussianDiffusionModel instance 
        :param ema: exponential moving average unet for sampling
        :param args: arguments dictionary form test_args
    """
    print("computing psnr")
    out = diffusion.forward_backward(ema, x, see_whole_sequence=None, t_distance=args["T"] // 2)
    # 24/01/2025 RM TODO maybe compute example images here on the test-set
    psnr = PSNR(out, x)
    return psnr

def compute_loss(x, diffusion, model, args):
    """
        :param analyzing_dataset_loader: cycle(dataloader) instance for evaluation
        :param diffusion: GaussianDiffusionModel instance 
        :param model: original unet for VLB calc
        :param args: arguments dictionary form test_args
    """
    print("computing loss")
    loss, estimates = diffusion.p_loss(model, x, args)
    return loss


def main():
    """
        Load arguments, run training and testing functions, then remove checkpoint directory
    :return:
    """

    
    check_args()
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

    # get the images into a dataset loader
    analyzing_dataset = init_dataset("./", args)
    analyzing_dataset_loader = init_dataset_loader(analyzing_dataset, args)

    # get one image for analysis
    data = next(analyzing_dataset_loader)
    x = data["image"]
    x = x.to(device)
    # create video TODO get running
    create_video(analyzing_dataset_loader, diffusion, ema, args)

    create_image(x, diffusion, unet, ema, args)

    create_final_image(x, diffusion, unet, ema, args)
    
    ### compute VLB performance indicator
    vlb = compute_vlb(x, diffusion, unet, args)

    # compute Peak Signal to Noise ratio (PSNR)
    psnr = compute_psnr(x, diffusion, ema, args)

    # compute loss
    loss = compute_loss(x, diffusion, unet, args)

    print(f"Variational lowe bound:{vlb}")
    print(f"Peak signal to noise ratio (PSNR):{psnr}")
    print(f"Loss:{loss}")

    #### TODO ????
    """
     output_250 = diff.forward_backward(
                unet, img[slice, ...].reshape(1, 1, *args["img_size"]),
                see_whole_sequence="whole",
                # t_distance=5, denoise_fn=args["noise_fn"]
                t_distance=250, denoise_fn=args["noise_fn"]
                )

        output_250_images, mse_threshold_250 = make_prediction(
                img[slice, ...].reshape(1, 1, *args["img_size"]), output_250[-1].to(device),
                img_mask[slice, ...].reshape(1, 1, *args["img_size"]), output_250[251 // 2].to(device)
                )

    """
    

if __name__ == '__main__':
    # get the device 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main()


