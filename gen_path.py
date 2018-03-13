import os
import os.path as osp


def gen_path_list(data_folder):
    path_list = []
    for dir_name in sorted(os.listdir(data_folder)):
        if dir_name[0] == '.':
            continue
        for filename in sorted(os.listdir(osp.join(data_folder, dir_name))):
            if filename[0] == '.':
                continue
            path_list.append(osp.join(dir_name, filename))
    return path_list