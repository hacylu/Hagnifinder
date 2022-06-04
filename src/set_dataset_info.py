import os
import glob

dataset_txt = 'E:\\code\\Hagnifinder\\dataset\\Hagni40.txt'
path = 'E:\\code\\Hagnifinder\\Hagni40'  # Root directory


def text_save(filename, data, strr):
    filenamelist = data[0].split('\\')
    file = open(filename, strr)
    for i in range(len(data)):
        s = str(data[i]) + ' ' + filenamelist[-2] + ' ' + filenamelist[-3] + '\n'
        file.write(s)
    file.close()


def get_image_info(path, savedir):
    tarlist = os.listdir(path)
    categrylist = os.listdir(os.path.join(path + '\\' + tarlist[0]))
    for i in range(len(tarlist)):
        for j in range(len(categrylist)):
            imglist = glob.glob(os.path.join(path + '\\' + tarlist[i] + '\\' + categrylist[j] + '\\*.png'))
            if i == 0 and j == 0:
                strr = 'w'
            else:
                strr = 'a'
            text_save(savedir, imglist, strr)


if __name__ == "__main__":
    get_image_info(path, dataset_txt)
