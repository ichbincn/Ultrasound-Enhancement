import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import glob

def calculate_psnr_ssim(img1, img2):
    """计算单对图片的 PSNR 和 SSIM 值"""
    psnr_value = psnr(img1, img2)

    # 如果是多通道图像，计算时指定通道
    if len(img1.shape) == 3:
        # 根据图像的最小尺寸来动态调整 win_size
        min_dim = min(img1.shape[0], img1.shape[1])
        win_size = min(7, min_dim // 2 * 2 - 1)  # 确保 win_size 为奇数且不超过图像尺寸
        ssim_value = ssim(img1, img2, channel_axis=-1, win_size=win_size)
    else:
        ssim_value = ssim(img1, img2)  # 灰度图像不需要指定 channel_axis

    return psnr_value, ssim_value

def load_images(image_folder1, image_folder2):
    """加载文件夹中的图片，支持png和jpg格式"""
    image_files1 = sorted(glob.glob(f"{image_folder1}/*.[pj][pn]g"))
    image_files2 = sorted(glob.glob(f"{image_folder2}/*.[pj][pn]g"))

    images1 = [cv2.imread(img) for img in image_files1]
    images2 = [cv2.imread(img) for img in image_files2]

    return images1, images2

def resize_images_to_fixed_size(images1, images2, target_size=(512, 512)):
    """将两组图片调整为固定大小"""
    resized_images1 = [cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR) for img in images1]
    resized_images2 = [cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR) for img in images2]

    return resized_images1, resized_images2

def calculate_average_psnr_ssim(images1, images2):
    """计算所有图片的平均 PSNR 和 SSIM 值"""
    total_psnr = 0
    total_ssim = 0
    num_images = len(images1)

    for img1, img2 in zip(images1, images2):
        psnr_value, ssim_value = calculate_psnr_ssim(img1, img2)
        total_psnr += psnr_value
        total_ssim += ssim_value

    avg_psnr = total_psnr / num_images
    avg_ssim = total_ssim / num_images

    return avg_psnr, avg_ssim

# 指定两个文件夹，分别存放原始图像和处理后的图像
folder1 = "/Users/chennuo/Documents/专业上的东西/超声2.0/代码/结果/papertest"
folder2 = "/Users/chennuo/Documents/专业上的东西/超声2.0/代码/结果/paperHH/消融实验"

# 加载图片
images1, images2 = load_images(folder1, folder2)

# 调整图片尺寸为 512x512
images1_resized, images2_resized = resize_images_to_fixed_size(images1, images2, target_size=(512, 512))

# 计算平均PSNR和SSIM
avg_psnr, avg_ssim = calculate_average_psnr_ssim(images1_resized, images2_resized)

print(f"Average PSNR: {avg_psnr}")
print(f"Average SSIM: {avg_ssim}")
