"""
此文件用于读取标注信息的csv文件，提取出数据-标签对，保存为字典
删去标签形状与原始数据形状不同的数据
最终dict格式为：[{data: data_path, cancer: cancer_path, lymph: lymph_path}, ...]
"""


import os

# 图像.dcm文件读取
import pydicom
# 标签.nii.gz文件读取
import nibabel as nib


# 读入csv文件
def read_csv(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    del lines[0]  # 删除第一行标题
    csv_ls = []
    for line in lines:
        line = line.strip('\n')  # 去除\n
        line = line.split(',')  # 以','分割
        csv_ls.append(line)
    return csv_ls

# 提取list中的数据-标签对，用字典保存
def csv2dict(csv_ls):
    dirpath = "/miying/project_data/"  # 数据所在目录
    dict_ls = []
    for i in range(len(csv_ls) // 2):
        dc = {}
        dc["data"] = os.path.join(dirpath, csv_ls[i * 2][0], csv_ls[i * 2][1], csv_ls[i * 2][2])  # 原始数据
        dc["cancer"] = os.path.join(dirpath, csv_ls[i * 2][4])  # 肿瘤标记
        dc["lymph"] = os.path.join(dirpath, csv_ls[i * 2 + 1][4])  # 异常淋巴结标记
        dict_ls.append(dc)
    return dict_ls

# 筛选出有问题的数据
def data_select(dict_ls):
    final_dict_ls = dict_ls  # 复制dict_ls
    j = 0
    del_ls = []
    for i in range(len(dict_ls)):
        try:
            # 读取数据-标签的形状
            cancer = nib.load(dict_ls[i]["cancer"]).shape[2]
            lymph = nib.load(dict_ls[i]["lymph"]).shape[2]
            data_len = len(os.listdir(dict_ls[i]["data"]))
            # 判断形状是否一致
            if cancer != data_len or lymph != data_len:
                del_ls.append(i)  # 记录有问题的数据
                print(f"delete data num{i}")
        # 读取失败的数据
        except:
            del_ls.append(i)  # 记录有问题的数据
            print(f"error, del data num {i}")
    # 从后往前删除，防止删除后索引变化
    del_ls.reverse()
    for i in del_ls:
        del final_dict_ls[i]
    return final_dict_ls
        
# 检查数据是否正确
def examine(final_dict_ls):
    for i in range(len(final_dict_ls)):
        cancer = nib.load(final_dict_ls[i]["cancer"]).shape[2]
        lymph = nib.load(final_dict_ls[i]["lymph"]).shape[2]
        data_len = len(os.listdir(final_dict_ls[i]["data"]))
        if cancer != data_len or lymph != data_len:
            print("error in data_select, check the code")
            return False
    return True
    
    
    
if __name__ == "__main__":
    path = "./120056_标注信息.CSV"
    csv_ls = read_csv(path)
    dict_ls = csv2dict(csv_ls)
    final_dict_ls = data_select(dict_ls)
    if examine(final_dict_ls):
        # 保存dict
        with open('./final_dict_ls.txt', 'w') as f:
            f.write(str(final_dict_ls))
        # 读取dict
        with open('./final_dict_ls.txt', 'r') as f:
            final_dict_ls = eval(f.read())
        if type(final_dict_ls) == list:
            print("Done!")
        else:
            print("Error!")
