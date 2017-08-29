import os
import sys

directory = '/home/srq/Datasets/people-counting-1'
directory_out = '/home/srq/Datasets/people-counting-1/train-two'

if len(sys.argv) != 2:
    print("Error")

    sys.exit()

files_list = sys.argv[1]

print(directory + '/' + files_list)

with open(directory + '/' + files_list) as f:
    for i in f:
        i = i[:-1]
        if not i.endswith('.jpg'):
            continue
        print(os.path.splitext(i)[0])
        file_name = os.path.splitext(i)[0]
        path = directory+'/'+ file_name
        image_path = path + '.jpg'
        gt_path = directory + '/' + file_name + '_gt.mat.csv'
        out_suffx = directory_out + '/' + file_name
        if not os.path.exists(out_suffx):
            os.mkdir(out_suffx)
        print(image_path, gt_path, out_suffx)
        command = '/home/srq/Projects/SwitchCounting/density_map/cmake-build-debug/DensityMap'
        os.system(command + ' ' + image_path + ' ' + gt_path + ' ' + out_suffx)