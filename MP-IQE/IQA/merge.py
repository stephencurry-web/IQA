# 文件路径
file_paths = ["/home/pws/IQA/LIQE-main/IQA_Database/kadid10k/splits2/1/kadid10k_train_clip.txt", 
              "/home/pws/IQA/LIQE-main/IQA_Database/kadid10k/splits2/1/kadid10k_val_clip.txt", 
              "/home/pws/IQA/LIQE-main/IQA_Database/kadid10k/splits2/1/kadid10k_test_clip.txt"]

# 合并后的文件路径
merged_file_path = "/home/pws/IQA/global_local/IQA/kadid10k_all_clip.txt"

# 打开合并后的文件，准备写入合并后的内容
with open(merged_file_path, "w") as merged_file:
    # 遍历每个文件
    for file_path in file_paths:
        # 打开文件
        with open(file_path, "r") as file:
            # 读取文件内容
            content = file.read()
            # 将内容写入合并后的文件
            merged_file.write(content)
