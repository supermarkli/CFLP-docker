#!/usr/bin/env python3
"""
将多个PDF图片添加子标题后拼接成一个PDF
布局：第一行两个图(a)(b)并排，第二行一个图(c)居中
"""

import fitz  # PyMuPDF
from PIL import Image, ImageDraw, ImageFont
import io
import os

# 配置
INPUT_FILES = [
    ("fig(5).pdf", "(a) 客户端注册与启动同步"),
    ("fig(6).pdf", "(b) 本地训练与安全聚合"),
    ("fig(7).pdf", "(c) 全局模型更新与迭代"),
]
OUTPUT_FILE = "fig_combined.pdf"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 样式配置
DPI = 300  # 渲染DPI
TITLE_FONT_SIZE = 100  # 子标题字号
TITLE_MARGIN_TOP = 30  # 标题与图片间距
TITLE_MARGIN_BOTTOM = 60  # 标题下方间距
PADDING = 80  # 图片间距
BACKGROUND_COLOR = (255, 255, 255)  # 白色背景


def pdf_to_image(pdf_path: str, dpi: int = 300) -> Image.Image:
    """将PDF第一页转换为PIL Image"""
    doc = fitz.open(pdf_path)
    page = doc[0]
    # 计算缩放矩阵
    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    
    # 转换为PIL Image
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    doc.close()
    return img


def get_font(size: int):
    """
    获取字体：优先使用当前目录下的 simsun.ttf
    """
    # 1. 最优先：直接加载当前目录下的 simsun.ttf
    # 这样你的代码本身就自带字体，去哪都能跑
    local_font_path = "simsun.ttf" 
    if os.path.exists(local_font_path):
        try:
            return ImageFont.truetype(local_font_path, size)
        except Exception as e:
            print(f"当前目录字体加载失败: {e}")


def add_title_to_image(img: Image.Image, title: str, font_size: int = 48) -> Image.Image:
    """在图片下方添加居中标题"""
    font = get_font(font_size)
    
    # 创建临时画布计算文字尺寸
    temp_draw = ImageDraw.Draw(img)
    bbox = temp_draw.textbbox((0, 0), title, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # 创建新图片，包含原图和标题区域
    new_height = img.height + TITLE_MARGIN_TOP + text_height + TITLE_MARGIN_BOTTOM
    new_img = Image.new("RGB", (img.width, new_height), BACKGROUND_COLOR)
    
    # 粘贴原图
    new_img.paste(img, (0, 0))
    
    # 绘制标题
    draw = ImageDraw.Draw(new_img)
    text_x = (img.width - text_width) // 2
    text_y = img.height + TITLE_MARGIN_TOP
    draw.text((text_x, text_y), title, font=font, fill=(0, 0, 0), stroke_width=2, stroke_fill=(0, 0, 0))
    
    return new_img


def create_combined_image(images: list[Image.Image]) -> Image.Image:
    """
    创建组合图片
    布局：第一行两个图并排，第二行一个图居中
    """
    img_a, img_b, img_c = images
    
    # 计算第一行尺寸（两图并排）
    row1_width = img_a.width + PADDING + img_b.width
    row1_height = max(img_a.height, img_b.height)
    
    # 计算第二行尺寸（单图）
    row2_width = img_c.width
    row2_height = img_c.height
    
    # 计算总画布尺寸
    total_width = max(row1_width, row2_width)
    total_height = row1_height + PADDING + row2_height
    
    # 创建画布
    canvas = Image.new("RGB", (total_width, total_height), BACKGROUND_COLOR)
    
    # 第一行：两图并排，整体居中
    row1_offset_x = (total_width - row1_width) // 2
    
    # 粘贴图a（左）
    y_a = (row1_height - img_a.height) // 2
    canvas.paste(img_a, (row1_offset_x, y_a))
    
    # 粘贴图b（右）
    y_b = (row1_height - img_b.height) // 2
    canvas.paste(img_b, (row1_offset_x + img_a.width + PADDING, y_b))
    
    # 第二行：图c居中
    x_c = (total_width - img_c.width) // 2
    y_c = row1_height + PADDING
    canvas.paste(img_c, (x_c, y_c))
    
    return canvas


def save_image_as_pdf(img: Image.Image, output_path: str):
    """将PIL Image保存为PDF"""
    # 转换为RGB模式（如果需要）
    if img.mode != "RGB":
        img = img.convert("RGB")
    
    # 保存为PDF
    img.save(output_path, "PDF", resolution=DPI)
    print(f"已保存: {output_path}")


def main():
    print("开始处理...")
    
    processed_images = []
    
    for filename, title in INPUT_FILES:
        filepath = os.path.join(SCRIPT_DIR, filename)
        print(f"处理: {filename} -> {title}")
        
        # PDF转图片
        img = pdf_to_image(filepath, DPI)
        print(f"  图片尺寸: {img.width} x {img.height}")
        
        # 添加标题
        img_with_title = add_title_to_image(img, title, TITLE_FONT_SIZE)
        processed_images.append(img_with_title)
    
    # 拼接图片
    print("拼接图片...")
    combined = create_combined_image(processed_images)
    print(f"组合图片尺寸: {combined.width} x {combined.height}")
    
    # 保存为PDF
    output_path = os.path.join(SCRIPT_DIR, OUTPUT_FILE)
    save_image_as_pdf(combined, output_path)
    
    print("完成!")


if __name__ == "__main__":
    main()
