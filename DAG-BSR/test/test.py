from option.DAGBSR.option_test import args
from model.DAGBSR.DAGBSR import DAGBSR
from utils import utility, degradation1
from data.srdataset import SRDataset
from torch.utils.data import DataLoader
import torch
import random
import os


def load_model(model, model_path, model_name):
    if os.path.isfile(model_path):
        print("Loading model", model_name, "from", model_path)
        checkpoint = torch.load(model_path)
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        print("Test model", model_name, "from checkpoint", args.start_epoch - 1)
    else:
        print("Model path does not exist:", model_path)


def main():
    if args.seed is not None:
        random.seed(args.seed)
    torch.manual_seed(args.seed)

    model_paths = [
        'DAGBSR/experiment/x4iso/model/model_0001.pth.tar',
    ]
    model_names = [
        'x4iso'
    ]

    models = [DAGBSR().cuda() for _ in range(len(model_paths))]

    for model, model_path, model_name in zip(models, model_paths, model_names):
        load_model(model, model_path, model_name)

    Test_List = ['Set5']
    sigmas = [0.0,3.6]
    for name in Test_List:
        dataset_test = SRDataset(args, name=name, train=True)
        dataloader_test = DataLoader(dataset=dataset_test, batch_size=1, shuffle=False, num_workers=4, pin_memory=True,
                                     drop_last=False)
        for i in range(0, 2):
            sigma = sigmas[i]
            if sigma == 0.0:
                degrade = degradation1.BicubicDegradation(args.scale)
            else:
                degrade = degradation1.IsoDegradation(sigma)
            print(f"Degradation parameters:")
            print(f"Sigma: {sigma}")
            model_list = [model]
            batch_deg_test(dataloader_test, model_list, args, degrade)


def batch_deg_test(test_loader, model_list, args, degrade):
    with torch.no_grad():
        test_psnr_list = [0] * len(model_list)
        test_ssim_list = [0] * len(model_list)

        for batch, (hr, _) in enumerate(test_loader):
            hr = hr.cuda(non_blocking=True)

            hr = crop_border_window(hr, args.scale, args.window_size)
            hr = hr.unsqueeze(1)
            lr = degrade(hr)

            hr = hr[:, 0, ...]
            lr = lr[:, 0, ...]

            hr = utility.quantize(hr, args.rgb_range)

            for i, model in enumerate(model_list):
                model.eval()
                sr = model(lr)
                sr = sr
                sr = utility.quantize(sr, args.rgb_range)

                test_psnr_list[i] += utility.calc_psnr(sr, hr, args.scale, args.rgb_range, benchmark=True)
                test_ssim_list[i] += utility.calc_ssim(sr, hr, args.scale, benchmark=True)
        for i in range(len(model_list)):
            print("{:.2f}/{:.4f}".format(test_psnr_list[i] / len(test_loader),
                                         test_ssim_list[i] / len(test_loader)))


def crop_border_test(img, scale):
    b, c, h, w = img.size()

    img = img[:, :, :int(h // scale * scale), :int(w // scale * scale)]

    return img


def crop_border_window(img, scale=2, window_size=16):
    _, _, h, w = img.size()
    img = img[:, :, :int(h // scale // window_size * scale * window_size),
          :int(w // scale // window_size * scale * window_size)]
    return img


if __name__ == '__main__':
    main()
