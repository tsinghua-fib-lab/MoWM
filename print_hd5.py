import h5py
import sys
import numpy as np

def print_hdf5_item(name, obj, indent_level=0):
    """
    递归地打印 HDF5 文件中的一个项目（组或数据集）。
    """
    indent = '    ' * indent_level  # 缩进
    if isinstance(obj, h5py.Group):
        print(f"{indent}📂 组 (Group): {name}")
        # 递归遍历组内的项目
        for key, val in obj.items():
            print_hdf5_item(key, val, indent_level + 1)
            
    elif isinstance(obj, h5py.Dataset):
        print(f"{indent}📄 数据集 (Dataset): {name}")
        print(f"{indent}   - 形状 (Shape): {obj.shape}")
        print(f"{indent}   - 类型 (Dtype): {obj.dtype}")
        # 可选：打印数据集的一小部分内容
        try:
            # 根据维度打印不同内容
            if obj.ndim == 0: # 标量
                 print(f"{indent}   - 内容 (Content): {obj[()]}")
            elif obj.ndim == 1:
                preview = obj[:5] # 打印前5个元素
                print(f"{indent}   - 内容预览 (Preview): {preview} ...")
            else:
                preview = obj[tuple(slice(0, 2) for _ in range(obj.ndim))] # 打印每个维度的前2个
                print(f"{indent}   - 内容预览 (Preview):\n{indent}     {np.array2string(preview, indent=len(indent) + 5)}")

        except TypeError:
             print(f"{indent}   - 内容 (Content): 不可直接预览的数据类型")
        except Exception as e:
            print(f"{indent}   - 预览时出错: {e}")

def explore_hdf5_file(file_path):
    """
    打开并浏览一个 HDF5 文件的结构。
    """
    try:
        with h5py.File(file_path, 'r') as f:
            print(f"--- 正在打印 '{file_path}' 的文件结构 ---")
            print_hdf5_item('/', f['/']) # 从根组开始
            print("--- 打印结束 ---")
            
    except FileNotFoundError:
        print(f"错误: 文件 '{file_path}' 未找到。")
    except Exception as e:
        print(f"读取文件时发生错误: {e}")

# --- 使用示例 ---
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("使用方法: python print_hdf5_structure.py <你的h5文件路径>")
        # 创建一个简单的 HDF5 文件用于测试
        test_file = "test_data.h5"
        print(f"\n未提供文件路径，将创建一个测试文件 '{test_file}' 并打印。")
        with h5py.File(test_file, 'w') as f:
            f.create_dataset('scalar_data', data=123.45)
            g1 = f.create_group('group1')
            g1.create_dataset('vector_data', data=np.arange(10))
            g2 = g1.create_group('nested_group')
            g2.create_dataset('matrix_data', data=np.random.rand(3, 4))
        explore_hdf5_file(test_file)
    else:
        file_to_print = sys.argv[1]
        explore_hdf5_file(file_to_print)