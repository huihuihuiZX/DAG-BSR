import math
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from utils import diffjpeg


# Blur
def cal_sigma(sig_x, sig_y, radians):
    sig_x = sig_x.view(-1, 1, 1)
    sig_y = sig_y.view(-1, 1, 1)
    radians = radians.view(-1, 1, 1)

    D = torch.cat([F.pad(sig_x ** 2, [0, 1, 0, 0]), F.pad(sig_y ** 2, [1, 0, 0, 0])], 1)
    U = torch.cat([torch.cat([radians.cos(), -radians.sin()], 2),
                   torch.cat([radians.sin(), radians.cos()], 2)], 1)
    sigma = torch.bmm(U, torch.bmm(D, U.transpose(1, 2)))

    return sigma


def isotropic_gaussian_kernel(kernel_size, sigma):
    ax = torch.arange(kernel_size).float().cuda() - kernel_size // 2
    xx = ax.repeat(kernel_size).view(1, kernel_size, kernel_size)
    yy = ax.repeat_interleave(kernel_size).view(1, kernel_size, kernel_size)
    kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2. * sigma.view(-1, 1, 1) ** 2))

    return kernel / kernel.sum([1, 2], keepdim=True)


def anisotropic_gaussian_kernel(kernel_size, covar):
    ax = torch.arange(kernel_size).float().cuda() - kernel_size // 2

    xx = ax.repeat(kernel_size).view(1, kernel_size, kernel_size)
    yy = ax.repeat_interleave(kernel_size).view(1, kernel_size, kernel_size)
    xy = torch.stack([xx, yy], -1).view(1, -1, 2)

    covar = covar.cpu()
    inverse_sigma = torch.inverse(covar).cuda()
    # inverse_sigma = np.linalg.inv(covar)
    # inverse_sigma = torch.tensor(inverse_sigma).cuda()
    kernel = torch.exp(- 0.5 * (torch.bmm(xy, inverse_sigma) * xy).sum(2)).view(1, kernel_size, kernel_size)

    return kernel / kernel.sum([1, 2], keepdim=True)


def random_isotropic_gaussian_kernel(kernel_size=21, sigma_min=0.2, sigma_max=4.0):
    x = torch.rand(1).cuda() * (sigma_max - sigma_min) + sigma_min
    kernel = isotropic_gaussian_kernel(kernel_size, x)
    return kernel


def random_anisotropic_gaussian_kernel(kernel_size=21, sigma_min=0.2, sigma_max=4.0):
    theta = torch.rand(1).cuda() * math.pi
    sigma_x = torch.rand(1).cuda() * (sigma_max - sigma_min) + sigma_min
    sigma_y = torch.rand(1).cuda() * (sigma_max - sigma_min) + sigma_min
    # print(sigma_x, sigma_y)
    covar = cal_sigma(sigma_x, sigma_y, theta)
    kernel = anisotropic_gaussian_kernel(kernel_size, covar)
    return kernel


# generalized gaussian kernel
def isotropic_generalized_kernel(kernel_size, sigma, beta):
    ax = torch.arange(kernel_size).float().cuda() - kernel_size // 2
    xx = ax.repeat(kernel_size).view(1, kernel_size, kernel_size)
    yy = ax.repeat_interleave(kernel_size).view(1, kernel_size, kernel_size)

    kernel = torch.exp(-0.5 * torch.pow((xx ** 2 + yy ** 2) / (1. * sigma.view(-1, 1, 1) ** 2), beta))

    return kernel / kernel.sum([1, 2], keepdim=True)


def anisotropic_generalized_kernel(kernel_size, covar, beta):
    ax = torch.arange(kernel_size).float().cuda() - kernel_size // 2

    xx = ax.repeat(kernel_size).view(1, kernel_size, kernel_size)
    yy = ax.repeat_interleave(kernel_size).view(1, kernel_size, kernel_size)
    xy = torch.stack([xx, yy], -1).view(1, -1, 2)

    covar = covar.cpu()
    inverse_sigma = torch.inverse(covar).cuda()
    kernel = torch.exp(- 0.5 * torch.pow((torch.bmm(xy, inverse_sigma) * xy).sum(2), beta)).view(1, kernel_size,
                                                                                                 kernel_size)

    return kernel / kernel.sum([1, 2], keepdim=True)


def random_isotropic_generalized_kernel(kernel_size=21, sigma_min=0.2, sigma_max=4.0, beta_min=0.5, beta_max=4.0):
    sigma = torch.rand(1).cuda() * (sigma_max - sigma_min) + sigma_min
    beta = torch.rand(1).cuda() * (beta_max - beta_min) + beta_min
    # print(sigma)
    kernel = isotropic_generalized_kernel(kernel_size, sigma, beta)
    return kernel


def random_anisotropic_generalized_kernel(kernel_size=21, sigma_min=0.2, sigma_max=4.0, beta_min=0.5, beta_max=4.0):
    theta = torch.rand(1).cuda() * math.pi
    sigma_x = torch.rand(1).cuda() * (sigma_max - sigma_min) + sigma_min
    sigma_y = torch.rand(1).cuda() * (sigma_max - sigma_min) + sigma_min
    beta = torch.rand(1).cuda() * (beta_max - beta_min) + beta_min
    # print(sigma_x, sigma_y)
    covar = cal_sigma(sigma_x, sigma_y, theta)
    kernel = anisotropic_generalized_kernel(kernel_size, covar, beta)
    return kernel


# plateau kernel
def isotropic_plateau_kernel(kernel_size, sigma, beta):
    ax = torch.arange(kernel_size).float().cuda() - kernel_size // 2
    xx = ax.repeat(kernel_size).view(1, kernel_size, kernel_size)
    yy = ax.repeat_interleave(kernel_size).view(1, kernel_size, kernel_size)

    kernel = 1.0 / (torch.pow((xx ** 2 + yy ** 2) / (1. * sigma.view(-1, 1, 1) ** 2), beta) + 1)

    return kernel / kernel.sum([1, 2], keepdim=True)


def random_isotropic_plateau_kernel(kernel_size=21, sigma_min=0.2, sigma_max=3.0, beta_min=1.0, beta_max=2.0):
    sigma = torch.rand(1).cuda() * (sigma_max - sigma_min) + sigma_min
    beta = torch.rand(1).cuda() * (beta_max - beta_min) + beta_min
    # print(sigma)
    kernel = isotropic_plateau_kernel(kernel_size, sigma, beta)
    return kernel


def anisotropic_plateau_kernel(kernel_size, covar, beta):
    ax = torch.arange(kernel_size).float().cuda() - kernel_size // 2

    xx = ax.repeat(kernel_size).view(1, kernel_size, kernel_size)
    yy = ax.repeat_interleave(kernel_size).view(1, kernel_size, kernel_size)
    xy = torch.stack([xx, yy], -1).view(1, -1, 2)

    covar = covar.cpu()
    inverse_sigma = torch.inverse(covar).cuda()
    kernel = (1.0 / (torch.pow((torch.bmm(xy, inverse_sigma) * xy).sum(2), beta) + 1)).view(1, kernel_size, kernel_size)

    return kernel / kernel.sum([1, 2], keepdim=True)


def random_anisotropic_plateau_kernel(kernel_size=21, sigma_min=0.2, sigma_max=3.0, beta_min=1.0, beta_max=2.0):
    theta = torch.rand(1).cuda() * math.pi
    sigma_x = torch.rand(1).cuda() * (sigma_max - sigma_min) + sigma_min
    sigma_y = torch.rand(1).cuda() * (sigma_max - sigma_min) + sigma_min
    beta = torch.rand(1).cuda() * (beta_max - beta_min) + beta_min
    # print(sigma_x, sigma_y)
    covar = cal_sigma(sigma_x, sigma_y, theta)
    kernel = anisotropic_plateau_kernel(kernel_size, covar, beta)
    return kernel


# total random gaussian kernel
def generate_random_kernel(kernel_size=21, blur_type='iso_gaussian', sigma_min=0.2, sigma_max=4.0,
                           betag_min=0.5, betag_max=4.0, betap_min=1.0, betap_max=2.0, ):
    if blur_type == 'iso_gaussian':
        return random_isotropic_gaussian_kernel(kernel_size=kernel_size, sigma_min=sigma_min, sigma_max=sigma_max)
    elif blur_type == 'aniso_gaussian':
        return random_anisotropic_gaussian_kernel(kernel_size=kernel_size, sigma_min=sigma_min, sigma_max=sigma_max)
    elif blur_type == 'iso_generalized':
        return random_isotropic_generalized_kernel(kernel_size=kernel_size, sigma_min=sigma_min, sigma_max=sigma_max,
                                                   beta_min=betag_min, beta_max=betag_max)
    elif blur_type == 'aniso_generalized':
        return random_anisotropic_generalized_kernel(kernel_size=kernel_size, sigma_min=sigma_min, sigma_max=sigma_max,
                                                     beta_min=betag_min, beta_max=betag_max)
    elif blur_type == 'iso_plateau':
        return random_isotropic_plateau_kernel(kernel_size=kernel_size, sigma_min=sigma_min, sigma_max=sigma_max,
                                               beta_min=betap_min, beta_max=betap_max)
    elif blur_type == 'aniso_plateau':
        return random_anisotropic_plateau_kernel(kernel_size=kernel_size, sigma_min=sigma_min, sigma_max=sigma_max,
                                                 beta_min=betap_min, beta_max=betap_max)


class BatchRandonKernel:
    def __init__(self, blur_type_list, blur_type_prob, kernel_size=21, sigma_min=0.2, sigma_max=4.0,
                 betag_min=0.5, betag_max=4.0, betap_min=1.0, betap_max=2.0):
        super(BatchRandonKernel, self).__init__()
        self.kernel_size = kernel_size
        self.blur_type_list = blur_type_list
        self.blur_type_prob = blur_type_prob

        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.betag_min = betag_min
        self.betag_max = betag_max
        self.betap_min = betap_min
        self.betap_max = betap_max

    def __call__(self, batch):
        batch_kernel = torch.zeros(batch, self.kernel_size, self.kernel_size).float().cuda()
        for i in range(batch):
            # random.seed(time.time())
            blur_type = random.choices(self.blur_type_list, self.blur_type_prob)[0]
            # print("blur_type", blur_type)
            kernel_i = generate_random_kernel(kernel_size=self.kernel_size, blur_type=blur_type,
                                              sigma_min=self.sigma_min, sigma_max=self.sigma_max,
                                              betag_min=self.betag_min, betag_max=self.betag_max,
                                              betap_min=self.betag_min, betap_max=self.betap_max, )
            batch_kernel[i, :, ...] = kernel_i
        return batch_kernel


class BatchBlur(nn.Module):
    def __init__(self, kernel_size=21):
        super(BatchBlur, self).__init__()
        self.kernel_size = kernel_size
        if kernel_size % 2 == 1:
            self.pad = nn.ReflectionPad2d(kernel_size // 2)
        else:
            self.pad = nn.ReflectionPad2d(
                (kernel_size // 2, kernel_size // 2 - 1, kernel_size // 2, kernel_size // 2 - 1))

    def forward(self, input, kernel):
        B, C, H, W = input.size()
        input_pad = self.pad(input)
        H_p, W_p = input_pad.size()[-2:]

        if len(kernel.size()) == 2:
            input_CBHW = input_pad.view((C * B, 1, H_p, W_p))
            kernel = kernel.contiguous().view((1, 1, self.kernel_size, self.kernel_size))

            return F.conv2d(input_CBHW, kernel, padding=0).view((B, C, H, W))
        else:
            input_CBHW = input_pad.view((1, C * B, H_p, W_p))
            kernel = kernel.contiguous().view((B, 1, self.kernel_size, self.kernel_size))
            kernel = kernel.repeat(1, C, 1, 1).view((B * C, 1, self.kernel_size, self.kernel_size))

            return F.conv2d(input_CBHW, kernel, groups=B * C).view((B, C, H, W))


# DownSample
class Bicubic(nn.Module):
    def __init__(self):
        super(Bicubic, self).__init__()

    def cubic(self, x):
        absx = torch.abs(x)
        absx2 = torch.abs(x) * torch.abs(x)
        absx3 = torch.abs(x) * torch.abs(x) * torch.abs(x)

        condition1 = (absx <= 1).to(torch.float32)
        condition2 = ((1 < absx) & (absx <= 2)).to(torch.float32)

        f = (1.5 * absx3 - 2.5 * absx2 + 1) * condition1 + (-0.5 * absx3 + 2.5 * absx2 - 4 * absx + 2) * condition2
        return f

    def contribute(self, in_size, out_size, scale):
        kernel_width = 4
        if scale < 1:
            kernel_width = 4 / scale
        x0 = torch.arange(start=1, end=out_size[0] + 1).to(torch.float32).cuda()
        x1 = torch.arange(start=1, end=out_size[1] + 1).to(torch.float32).cuda()

        u0 = x0 / scale + 0.5 * (1 - 1 / scale)
        u1 = x1 / scale + 0.5 * (1 - 1 / scale)

        left0 = torch.floor(u0 - kernel_width / 2)
        left1 = torch.floor(u1 - kernel_width / 2)

        P = np.ceil(kernel_width) + 2

        indice0 = left0.unsqueeze(1) + torch.arange(start=0, end=P).to(torch.float32).unsqueeze(0).cuda()
        indice1 = left1.unsqueeze(1) + torch.arange(start=0, end=P).to(torch.float32).unsqueeze(0).cuda()

        mid0 = u0.unsqueeze(1) - indice0.unsqueeze(0)
        mid1 = u1.unsqueeze(1) - indice1.unsqueeze(0)

        if scale < 1:
            weight0 = scale * self.cubic(mid0 * scale)
            weight1 = scale * self.cubic(mid1 * scale)
        else:
            weight0 = self.cubic(mid0)
            weight1 = self.cubic(mid1)

        weight0 = weight0 / (torch.sum(weight0, 2).unsqueeze(2))
        weight1 = weight1 / (torch.sum(weight1, 2).unsqueeze(2))

        indice0 = torch.min(torch.max(torch.FloatTensor([1]).cuda(), indice0),
                            torch.FloatTensor([in_size[0]]).cuda()).unsqueeze(0)
        indice1 = torch.min(torch.max(torch.FloatTensor([1]).cuda(), indice1),
                            torch.FloatTensor([in_size[1]]).cuda()).unsqueeze(0)

        kill0 = torch.eq(weight0, 0)[0][0]
        kill1 = torch.eq(weight1, 0)[0][0]

        weight0 = weight0[:, :, kill0 == 0]
        weight1 = weight1[:, :, kill1 == 0]

        indice0 = indice0[:, :, kill0 == 0]
        indice1 = indice1[:, :, kill1 == 0]

        return weight0, weight1, indice0, indice1

    def forward(self, input, scale=1 / 4):
        b, c, h, w = input.shape

        weight0, weight1, indice0, indice1 = self.contribute([h, w], [int(h * scale), int(w * scale)], scale)
        weight0 = weight0[0]
        weight1 = weight1[0]

        indice0 = indice0[0].long()
        indice1 = indice1[0].long()

        out = input[:, :, (indice0 - 1), :] * (weight0.unsqueeze(0).unsqueeze(1).unsqueeze(4))
        out = (torch.sum(out, dim=3))
        A = out.permute(0, 1, 3, 2)

        out = A[:, :, (indice1 - 1), :] * (weight1.unsqueeze(0).unsqueeze(1).unsqueeze(4))
        out = out.sum(3).permute(0, 1, 3, 2)

        return out


def generate_down_sample(hr_blured, down_sample_type, down_sample_scale):
    bicubic = Bicubic()
    if down_sample_type == 'nearest':
        return F.interpolate(hr_blured, scale_factor=1 / down_sample_scale, mode=down_sample_type,
                             recompute_scale_factor=True)
    elif down_sample_type == 'area':
        return F.interpolate(hr_blured, scale_factor=1 / down_sample_scale, mode=down_sample_type,
                             recompute_scale_factor=True)
    elif down_sample_type == 'bilinear':
        return F.interpolate(hr_blured, scale_factor=1 / down_sample_scale, mode=down_sample_type,
                             recompute_scale_factor=True, align_corners=True)
    elif down_sample_type == 'bicubic':
        return bicubic(hr_blured, scale=1 / down_sample_scale)


class BatchDownSample(nn.Module):
    def __init__(self, down_sample_list, down_sample_prob, down_sample_scale):
        super(BatchDownSample, self).__init__()
        self.down_sample_list = down_sample_list
        self.down_sample_prob = down_sample_prob
        self.down_sample_scale = down_sample_scale

    def __call__(self, hr_blured):
        B, C, W, H = hr_blured.shape
        W = int(W / self.down_sample_scale)
        H = int(H / self.down_sample_scale)
        lr_blured = torch.zeros(B, C, W, H).float().cuda()
        for i in range(0, B, 2):
            # random.seed(time.time())
            down_sample_type = random.choices(self.down_sample_list, self.down_sample_prob)[0]
            # print("down_sample_type is", down_sample_type)
            lr_blured[i:i + 2, :, ...] = generate_down_sample(hr_blured[i:i + 2, :, ...], down_sample_type,
                                                              self.down_sample_scale)
        return lr_blured


# Noise

def uniform_noise(noise_scale, width, height):
    noise = torch.rand(1, width, height).cuda()
    noise = noise * noise_scale
    return noise


def gaussian_noise(noise_scale, width, height, std_min=0, std_max=4):
    std = random.uniform(std_min, std_max)
    gaussian_noise = torch.normal(mean=0, std=std, size=(1, width, height)).cuda()
    gaussian_noise = gaussian_noise * noise_scale
    return gaussian_noise


def poisson_noise(noise_scale, width, height, lambda_min=1, lambda_max=3):
    lambda_poisson = lambda_min + torch.rand(1, width, height) * (lambda_max - lambda_min)
    poisson_noise = torch.poisson(lambda_poisson).cuda()
    poisson_noise = poisson_noise * noise_scale
    return poisson_noise


def generate_random_noise(width, height, noise_type, noise_scale, std_min, std_max, lambda_min, lambda_max):
    if noise_type == 'uniform_noise':
        return uniform_noise(noise_scale, width, height)
    elif noise_type == 'gaussian_noise':
        return gaussian_noise(noise_scale, width, height, std_min, std_max)
    elif noise_type == 'poisson_noise':
        return poisson_noise(noise_scale, width, height, lambda_min, lambda_max)


class BatchRandomNoise:
    def __init__(self, noise_list, noise_prob, noise_min, noise_max, std_min, std_max, lambda_min,
                 lambda_max):
        super(BatchRandomNoise, self).__init__()
        self.noise_list = noise_list
        self.noise_prob = noise_prob
        self.noise_min = noise_min
        self.noise_max = noise_max
        self.std_min = std_min
        self.std_max = std_max
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max

    def __call__(self, b, w, h):
        batch_noise = torch.zeros(b, w, h).float().cuda()
        for i in range(b):
            # random.seed(time.time())
            noise_type = random.choices(self.noise_list, self.noise_prob)[0]
            # print("Noise type: ", noise_type)
            # noise_scale = torch.rand(1).cuda() * (self.noise_max - self.noise_min) + self.noise_min
            noise_scale = self.noise_max
            # print("noise_scale", noise_scale)
            noise_i = generate_random_noise(w, h, noise_type, noise_scale,
                                            self.std_min, self.std_max,
                                            self.lambda_min, self.lambda_max,
                                            )
            batch_noise[i, :, ...] = noise_i
        return batch_noise


class BatchAddNoise(nn.Module):
    def __init__(self):
        super(BatchAddNoise, self).__init__()

    def __call__(self, lr_blured, n, c, batch_noise):
        lr_noised = torch.zeros_like(lr_blured).float().cuda()  # B, NC, W, H
        batch_noise = batch_noise.unsqueeze(1)  # 在nc维度增加一个维度，匹配尺寸用于加噪
        for i in range(n * c):
            lr_noised[:, i:i + 1, ...] = lr_blured[:, i:i + 1, ...].add_(batch_noise)
        return lr_noised


class BatchJpeg(nn.Module):
    def __init__(self, mode=255):
        super(BatchJpeg, self).__init__()
        self.mode = mode

    def __call__(self, lr_noised, jpeg_min, jpeg_max):
        B, C, W, H = lr_noised.shape
        if self.mode == 255:
            lr_noised = lr_noised / 255.0
        lr_jpeged = torch.zeros(B, C, W, H).float().cuda()
        jpeger = diffjpeg.DiffJPEG(differentiable=False).cuda()
        for i in range(0, B, 2):
            # random.seed(time.time())
            jpeg_q = torch.rand(1).cuda() * (jpeg_max - jpeg_min) + jpeg_min
            # print("jpeg_q: ", jpeg_q)
            lr_jpeged[i:i + 2, :, ...] = jpeger(lr_noised[i:i + 2, :, ...], jpeg_q)
        if self.mode == 255:
            lr_jpeged = lr_jpeged * 255.0
        return lr_jpeged


class MultiDegradation(object):
    def __init__(self,
                 scale,
                 kernel_size,
                 blur_type_list,
                 blur_type_prob,
                 sigma_min,
                 sigma_max,
                 betag_min,
                 betag_max,
                 betap_min,
                 betap_max,
                 down_sample_list,
                 down_sample_prob,
                 noise_list,
                 noise_prob,
                 noise_min,
                 noise_max,
                 std_min,
                 std_max,
                 lambda_min,
                 lambda_max,
                 jpeg_min,
                 jpeg_max,
                 ):
        self.kernel_size = kernel_size
        self.scale = scale
        self.jpeg_min = jpeg_min
        self.jpeg_max = jpeg_max

        self.gen_kernel = BatchRandonKernel(blur_type_list, blur_type_prob, kernel_size, sigma_min, sigma_max,
                                            betag_min, betag_max, betap_min, betap_max)
        self.blur = BatchBlur(kernel_size)
        self.down_sample = BatchDownSample(down_sample_list, down_sample_prob, scale)
        self.gen_noise = BatchRandomNoise(noise_list, noise_prob, noise_min, noise_max, std_min, std_max,
                                          lambda_min, lambda_max)
        self.noise = BatchAddNoise()
        self.jpeger = BatchJpeg(mode=255)

    def __call__(self, hr_tensor):
        with torch.no_grad():
            # Degradation
            B, N, C, H, W = hr_tensor.size()
            # print("hr tensor shape", hr_tensor.shape)
            # blur
            b_kernels = self.gen_kernel(B)  # B degradations size:[Batch, kernel_size, kernel_size]
            # print("kernel shape", b_kernels.shape)
            hr_tensor = hr_tensor.reshape(B, -1, H, W)  # ensure the channel n has same blur kernel
            hr_blured = self.blur(hr_tensor, b_kernels)  # .view(-1, C, H, W)
            # print("hr blured shape", hr_blured.shape)
            hr_blured = hr_blured.reshape(-1, C, H, W)
            # print("hr_blured2 shape", hr_blured.shape)
            # n通道整合 B*N, C, W, H; (hr[B, N, C, H, W]->hr_blured[B*N, C, H, W])
            # down_sample
            lr_downsampled = self.down_sample(hr_blured)
            # print("lr down sample shape", lr_downsampled.shape)
            # add noise
            lr_downsampled = lr_downsampled.reshape(B, -1, H // int(self.scale), W // int(self.scale))

            b_noises = self.gen_noise(B, H // int(self.scale), W // int(self.scale))
            lr_noised = self.noise(lr_downsampled, N, C, b_noises)
            lr_noised = lr_noised.reshape(-1, C, H // int(self.scale), W // int(self.scale))
            # print("lr noised shape", lr_noised.shape)
            # jpeg
            lr_jpeged = self.jpeger(lr_noised, self.jpeg_min, self.jpeg_max)
            # print("lr jpeged shape", lr_jpeged.shape)
            lr_jpeged = torch.clamp(lr_jpeged.round(), 0, 255)
            lr = lr_jpeged.reshape(B, N, C, H // int(self.scale), W // int(self.scale))
            # print("lr shape", lr.shape)
            return lr


class BatchBicubic(nn.Module):
    def __init__(self, down_sample_scale):
        super(BatchBicubic, self).__init__()
        self.down_sample_scale = down_sample_scale

    def __call__(self, hr):
        B, C, W, H = hr.shape
        W = int(W / self.down_sample_scale)
        H = int(H / self.down_sample_scale)
        lr = torch.zeros(B, C, W, H).float().cuda()
        for i in range(0, B, 2):
            # random.seed(time.time())
            down_sample_type = 'bicubic'
            # print("down_sample_type is", down_sample_type)
            lr[i:i + 2, :, ...] = generate_down_sample(hr[i:i + 2, :, ...], down_sample_type,
                                                       self.down_sample_scale)
        return lr


class deg_bicubic(object):
    def __init__(self, scale):
        self.scale = scale
        self.down_sample = BatchBicubic(scale)

    def __call__(self, hr_tensor):
        with torch.no_grad():
            # Degradation
            B, N, C, W, H = hr_tensor.size()
            hr = hr_tensor.reshape(-1, C, W, H)
            # down_sample
            lr_downsampled = self.down_sample(hr)
            lr = torch.clamp(lr_downsampled.round(), 0, 255)
            if self.scale > 1:
                lr = lr.reshape(B, N, C, W // int(self.scale), H // int(self.scale))
            else:
                lr = lr.reshape(B, N, C, int(W // self.scale), int(H // self.scale))
            # print("lr shape", lr.shape)
            return lr


class PcaEncoder(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.register_buffer("weight", weight)
        self.size = self.weight.size()

    def forward(self, batch_kernel):
        B, H, W = batch_kernel.size()  # [B, kernel size, kernel size]
        return torch.bmm(batch_kernel.view((B, 1, H * W)), self.weight.expand((B,) + self.size)).view((B, -1))


class DANDegradation(object):
    def __init__(self, args, pca_matrix):
        self.scale = args.scale
        self.kernel_size = args.kernel_size
        self.pca_matrix = pca_matrix
        self.code_length = args.code_length

        self.encoder = PcaEncoder(pca_matrix).cuda()
        self.gen_kernel = BatchRandonKernel(args.blur_type_list, args.blur_type_prob, args.kernel_size, args.sigma_min,
                                            args.sigma_max,
                                            args.betag_min, args.betag_max, args.betap_min, args.betap_max)
        self.blur = BatchBlur(args.kernel_size)
        self.down_sample = BatchDownSample(args.down_sample_list, args.down_sample_prob, args.scale)
        self.gen_noise = BatchRandomNoise(args.noise_list, args.noise_prob, args.noise_min, args.noise_max,
                                          args.std_min, args.std_max,
                                          args.lambda_min, args.lambda_max)
        self.noise = BatchAddNoise()

    def __call__(self, hr_tensor):
        with torch.no_grad():
            # Degradation
            B, N, C, H, W = hr_tensor.size()  # 32, 1, 3, 192, 192
            # print("hr tensor shape", hr_tensor.shape)
            # blur
            b_kernels = self.gen_kernel(B)  # B degradations size:[Batch, kernel_size, kernel_size]
            # print("kernel shape", b_kernels.shape)
            kernel_code = self.encoder(b_kernels)
            hr_tensor = hr_tensor.reshape(B, -1, H, W)  # ensure the channel n has same blur kernel
            hr_blured = self.blur(hr_tensor, b_kernels)  # .view(-1, C, H, W)
            # print("hr blured shape", hr_blured.shape)
            hr_blured = hr_blured.reshape(-1, C, H, W)
            # print("hr_blured2 shape", hr_blured.shape)
            # n通道整合 B*N, C, W, H; (hr[B, N, C, H, W]->hr_blured[B*N, C, H, W])
            # down_sample
            lr_downsampled = self.down_sample(hr_blured)
            # print("lr down sample shape", lr_downsampled.shape)
            # add noise
            lr_downsampled = lr_downsampled.reshape(B, -1, H // int(self.scale), W // int(self.scale))
            b_noises = self.gen_noise(B, H // int(self.scale), W // int(self.scale))
            lr_noised = self.noise(lr_downsampled, N, C, b_noises)
            lr_noised = lr_noised.reshape(-1, N, C, H // int(self.scale), W // int(self.scale))
            # print("lr noised shape", lr_noised.shape)
            lr = torch.clamp(lr_noised.round(), 0, 255)
            # jpeg
            # lr_jpeged = self.jpeger(lr_noised, self.jpeg_min, self.jpeg_max)
            # print("lr jpeged shape", lr_jpeged.shape)
            # lr_jpeged = torch.clamp(lr_jpeged.round(), 0, 255)
            # lr = lr_jpeged.reshape(B, N, C, H // int(self.scale), W // int(self.scale))
            # print("lr shape", lr.shape)
            return lr, kernel_code


class DANDegradationNF(object):
    def __init__(self, args, pca_matrix):
        self.scale = args.scale
        self.kernel_size = args.kernel_size
        self.pca_matrix = pca_matrix
        self.code_length = args.code_length

        self.encoder = PcaEncoder(pca_matrix).cuda()
        self.gen_kernel = BatchRandonKernel(args.blur_type_list, args.blur_type_prob, args.kernel_size, args.sigma_min,
                                            args.sigma_max,
                                            args.betag_min, args.betag_max, args.betap_min, args.betap_max)
        self.blur = BatchBlur(args.kernel_size)
        self.down_sample = BatchDownSample(args.down_sample_list, args.down_sample_prob, args.scale)

    def __call__(self, hr_tensor):
        with torch.no_grad():
            # Degradation
            B, N, C, H, W = hr_tensor.size()  # 32, 1, 3, 192, 192
            # print("hr tensor shape", hr_tensor.shape)
            # blur
            b_kernels = self.gen_kernel(B)  # B degradations size:[Batch, kernel_size, kernel_size]
            # print("kernel shape", b_kernels.shape)
            kernel_code = self.encoder(b_kernels)
            hr_tensor = hr_tensor.reshape(B, -1, H, W)  # ensure the channel n has same blur kernel
            hr_blured = self.blur(hr_tensor, b_kernels)  # .view(-1, C, H, W)
            # print("hr blured shape", hr_blured.shape)
            hr_blured = hr_blured.reshape(-1, C, H, W)
            # print("hr_blured2 shape", hr_blured.shape)
            # n通道整合 B*N, C, W, H; (hr[B, N, C, H, W]->hr_blured[B*N, C, H, W])
            # down_sample
            lr_downsampled = self.down_sample(hr_blured)
            # print("lr down sample shape", lr_downsampled.shape)
            # add noise
            lr_downsampled = lr_downsampled.reshape(B, -1, H // int(self.scale), W // int(self.scale))
            lr_downsampled = lr_downsampled.reshape(-1, N, C, H // int(self.scale), W // int(self.scale))
            # print("lr noised shape", lr_noised.shape)
            lr = torch.clamp(lr_downsampled.round(), 0, 255)

            return lr, kernel_code


class Degradation(object):
    def __init__(self, args):
        self.scale = args.scale
        self.kernel_size = args.kernel_size

        self.gen_kernel = BatchRandonKernel(args.blur_type_list, args.blur_type_prob, args.kernel_size, args.sigma_min,
                                            args.sigma_max,
                                            args.betag_min, args.betag_max, args.betap_min, args.betap_max)
        self.blur = BatchBlur(args.kernel_size)
        self.down_sample = BatchDownSample(args.down_sample_list, args.down_sample_prob, args.scale)
        self.gen_noise = BatchRandomNoise(args.noise_list, args.noise_prob, args.noise_min, args.noise_max,
                                          args.std_min, args.std_max,
                                          args.lambda_min, args.lambda_max)
        self.noise = BatchAddNoise()

    def __call__(self, hr_tensor):
        with torch.no_grad():
            # Degradation
            B, N, C, H, W = hr_tensor.size()  # 32, 1, 3, 192, 192
            # blur
            b_kernels = self.gen_kernel(B)  # B degradations size:[Batch, kernel_size, kernel_size]
            # print("kernel shape", b_kernels.shape)
            hr_tensor = hr_tensor.reshape(B, -1, H, W)  # ensure the channel n has same blur kernel
            hr_blured = self.blur(hr_tensor, b_kernels)  # .view(-1, C, H, W)
            # print("hr blured shape", hr_blured.shape)
            hr_blured = hr_blured.reshape(-1, C, H, W)
            # print("hr_blured2 shape", hr_blured.shape)
            # n通道整合 B*N, C, W, H; (hr[B, N, C, H, W]->hr_blured[B*N, C, H, W])
            # down_sample
            lr_downsampled = self.down_sample(hr_blured)
            # print("lr down sample shape", lr_downsampled.shape)
            # add noise
            lr_downsampled = lr_downsampled.reshape(B, -1, H // int(self.scale), W // int(self.scale))
            b_noises = self.gen_noise(B, H // int(self.scale), W // int(self.scale))
            lr_noised = self.noise(lr_downsampled, N, C, b_noises)
            lr_noised = lr_noised.reshape(-1, N, C, H // int(self.scale), W // int(self.scale))
            # print("lr noised shape", lr_noised.shape)
            lr = torch.clamp(lr_noised.round(), 0, 255)

            return lr


class G8Degradation(object):
    def __init__(self, args):
        self.scale = args.scale
        self.kernel_size = args.kernel_size

        self.gen_kernel = BatchRandonKernel(args.blur_type_list, args.blur_type_prob, args.kernel_size,
                                            args.sigma_min, args.sigma_max, args.betag_min, args.betag_max,
                                            args.betap_min, args.betap_max)
        self.blur = BatchBlur(args.kernel_size)
        self.down_sample = BatchDownSample(args.down_sample_list, args.down_sample_prob, args.scale)

    def __call__(self, hr_tensor):
        with torch.no_grad():
            # Degradation
            B, N, C, H, W = hr_tensor.size()  # 32, 1, 3, 192, 192
            # blur
            b_kernels = self.gen_kernel(B)  # B degradations size:[Batch, kernel_size, kernel_size]
            # print("kernel shape", b_kernels.shape)
            hr_tensor = hr_tensor.reshape(B, -1, H, W)  # ensure the channel n has same blur kernel
            hr_blured = self.blur(hr_tensor, b_kernels)  # .view(-1, C, H, W)
            # print("hr blured shape", hr_blured.shape)
            hr_blured = hr_blured.reshape(-1, C, H, W)
            # print("hr_blured2 shape", hr_blured.shape)
            # n通道整合 B*N, C, W, H; (hr[B, N, C, H, W]->hr_blured[B*N, C, H, W])
            # down_sample
            lr_downsampled = self.down_sample(hr_blured)
            # print("lr down sample shape", lr_downsampled.shape)
            lr_downsampled = lr_downsampled.reshape(B, -1, H // int(self.scale), W // int(self.scale))
            lr_downsampled = lr_downsampled.reshape(-1, N, C, H // int(self.scale), W // int(self.scale))
            # print("lr noised shape", lr_noised.shape)
            lr = torch.clamp(lr_downsampled.round(), 0, 255)

            return lr
