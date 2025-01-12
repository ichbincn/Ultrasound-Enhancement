import cv2
import os
import glob


def create_video_from_images(image_folder, output_video_path, fps=30, resolution=(512, 512)):
    """
    将指定文件夹中的图片组合成视频，并保存到指定路径
    :param image_folder: 存放图片的文件夹路径
    :param output_video_path: 输出视频的保存路径
    :param fps: 每秒帧数，默认30
    :param resolution: 视频分辨率，默认(512, 512)
    """
    # 获取文件夹中所有的图片文件，支持jpg和png
    image_files = sorted(glob.glob(os.path.join(image_folder, '*.[pj][pn]g')))

    # 如果没有找到图片，抛出异常
    if not image_files:
        raise ValueError(f"在文件夹 {image_folder} 中未找到任何 JPG 或 PNG 图片。")

    # 定义视频编码器并创建 VideoWriter 对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 指定编码格式
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, resolution)

    # 逐个读取图片，并将其写入视频文件中
    for img_file in image_files:
        img = cv2.imread(img_file)

        # 调整图片大小以适应指定的分辨率
        img_resized = cv2.resize(img, resolution)

        # 将图片写入视频
        video_writer.write(img_resized)

    # 释放 VideoWriter 对象，保存视频
    video_writer.release()
    print(f"视频已成功保存到: {output_video_path}")


# 定义输入的图片文件夹和输出的视频文件路径
image_folder = "/Users/chennuo/Documents/专业上的东西/超声2.0/代码/结果/Testdata/USCASE"
output_video_path = "/Users/chennuo/Documents/专业上的东西/超声2.0/代码/结果/Testdatavideo/USCASE/output_video.mp4"

# 调用函数生成视频
create_video_from_images(image_folder, output_video_path, fps=30, resolution=(512, 512))
