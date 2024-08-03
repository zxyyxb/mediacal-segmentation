"""
此文件用于转换数据格式并保存为.npy文件
"""

import numpy as np
import preprocess

def main():
    with open("./final_dict_ls.txt", "r") as f:
        final_dict_ls = eval(f.read())

    save_path = "/home/tione/notebook/model/data/monai/"

    i = 0
    for dc in final_dict_ls:
        data_path = dc["data"]
        cancer_path = dc["cancer"]
        data_array = preprocess.dcm2array(data_path)
        cancer_array = preprocess.nii2array(cancer_path)
        if data_array.shape[0] < 70:
            if data_array.shape[0] < 65:
                data_array = np.pad(data_array, ((0, 65), (0, 0), (0, 0)), "reflect")
                cancer_array = np.pad(cancer_array, ((0, 65), (0, 0), (0, 0)), "reflect")
            data_array = data_array[:65]
            cancer_array = cancer_array[:65]
        else:
            data_array = data_array[10:75]
            cancer_array = cancer_array[10:75]
        try:
            np.save(save_path + "img/" + str(i) + ".npy", data_array.reshape(1, 65, 512, 512))
            np.save(save_path + "seg/" + str(i) + ".npy", cancer_array.reshape(1, 65, 512, 512))
            i += 1
        except:
            print("error, skip the image")
            continue
        
    print("Done!")
    
if __name__ == "__main__":
    main()
