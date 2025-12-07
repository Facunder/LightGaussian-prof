from PIL import Image
import os

def downsample_image(input_path, output_path, scale_factor):
    """
    对 PNG 图片进行像素化/降采样处理，并保持最终图片大小不变。

    :param input_path: 输入 PNG 文件的路径
    :param output_path: 输出 PNG 文件的路径
    :param scale_factor: 降采样倍数，支持 4 (4x4 块) 或 2 (2x1 水平条)
    """
    # 检查文件是否存在
    if not os.path.exists(input_path):
        print(f"错误：找不到文件 {input_path}")
        return

    # 检查倍数是否有效
    if scale_factor not in [2, 4]:
        print("错误：降采样倍数必须是 2 或 4。")
        return

    try:
        # 打开图片
        img = Image.open(input_path).convert("RGB") # 转换为 RGB 模式，便于处理
        width, height = img.size
        
        # 创建一个新的、与原图大小相同的空白图片
        new_img = Image.new("RGB", (width, height))

        # --- 4x 降采样 (2x2 像素块) ---
        if scale_factor == 4:
            print("正在进行 2x2 块状降采样...")
            # 遍历 $4 \times 4$ 的网格
            for x in range(0, width, 2):
                for y in range(0, height, 2):
                    # 获取当前 $4 \times 4$ 网格的左上角像素颜色
                    # 使用 getpixel() 获取颜色元组 (R, G, B)
                    # 为了安全，这里使用 min() 确保索引不会超出边界，但由于步长为 4，这通常不是问题
                    
                    # 获取左上角像素颜色
                    try:
                        color = img.getpixel((x, y))
                    except IndexError:
                         # 如果 x 或 y 刚好是边界，跳过
                        continue

                    # 将这个 $4 \times 4$ 网格内的所有像素都设置为这个颜色
                    for i in range(x, min(x + 2, width)):
                        for j in range(y, min(y + 2, height)):
                            new_img.putpixel((i, j), color)

        # --- 2x 降采样 (2x1 水平条) ---
        elif scale_factor == 2:
            print("正在进行 2x1 水平降采样...")
            # 遍历图片，步长为 2 (只在 x 轴上)
            for x in range(0, width, 2):
                for y in range(height):
                    # 获取当前 $2 \times 1$ 区域的左边像素颜色 (即 (x, y))
                    try:
                        color = img.getpixel((x, y))
                    except IndexError:
                        continue # 如果 x 刚好是边界，跳过

                    # 将 (x, y) 和 (x+1, y) 的像素都设置为这个颜色
                    new_img.putpixel((x, y), color)
                    # 检查 x+1 是否还在图片宽度范围内
                    if x + 1 < width:
                        new_img.putpixel((x + 1, y), color)

        # 保存结果
        new_img.save(output_path, "PNG")
        print(f"✅ 处理完成！图片已保存到 {output_path}")

    except Exception as e:
        print(f"处理图片时发生错误: {e}")

# --- 用户输入和调用示例 ---

if __name__ == "__main__":
    # 1. 设定输入和输出路径
    input_file = "/root/autodl-tmp/LightGaussianOutput/lego_0.36/test_StreamingSort_noInterp_fullSortPer10frame_fp8/ours_35000/renders/00005.png" # 替换为你的输入 PNG 文件名
    
    # 2. 设定降采样倍数
    # 请选择 4 (4x4 块状) 或 2 (2x1 水平条状)
    scale = 2 

    # 3. 设定输出路径
    output_file = f"/root/autodl-tmp/LightGaussianOutput/lego_0.36/output_{scale}x_downsampled.png"

    # --- 运行函数 ---
    downsample_image(input_file, output_file, scale)

    # 💡 提示：如果你想尝试另一种倍数，可以修改 'scale' 变量并重新运行脚本。
    # 示例：
    # scale = 2
    # output_file = f"output_{scale}x_downsampled.png"
    # downsample_image(input_file, output_file, scale)