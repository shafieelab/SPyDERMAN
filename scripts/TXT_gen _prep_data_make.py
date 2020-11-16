import os
import random

random.seed(1000)

labels_virus_bubble = ["Negative", ]


def dataset_gen_txt(type):

    datasrt_name = type

    data_dir_path = '../Data/' + datasrt_name + '/'

    data_dir = os.listdir(data_dir_path)

    print(len(data_dir))
    for classes in labels_virus_bubble:

        print()

        images_dir = os.listdir(data_dir_path + classes)
        random.shuffle(images_dir)
        # random.Random(4).shuffle(images_dir)
        # images_dir = images_dir[0:40]

        #images_dir = images_dir[int(len(images_dir) * 0): int(len(images_dir) * 0.9)]
        #images_dir = images_dir[int(len(images_dir) * 0): int(len(images_dir) * 0.8)]
        images_dir = images_dir[int(len(images_dir) * 0): int(len(images_dir) * 1)]
        #images_dir = images_dir[int(len(images_dir) * 0.8): int(len(images_dir) * 1)]
        print(len(images_dir))

        for img_name in images_dir:
            # print(img_name,classes)
            if ".jpg" in img_name:
                with open('../Data/' + datasrt_name + "/" + "mb_" + "test.txt", 'a') as the_file:
                    # with open('../data/sd1/val.txt', 'a') as the_file:
                    # the_file.write(data_dir_path+img_name+" "+img_name+'\n')
                    the_file.write(
                        data_dir_path + classes + "/" + img_name + " " + str(labels_virus_bubble.index(classes)) + '\n')


def gen_target(type):
    datasrt_name = type

    data_dir_path = '../Data/Test/' + datasrt_name + '/Negative_Tile/'
    #data_dir_path = '../Data/GAN_Target/'

    data_dir = os.listdir(data_dir_path)

    print(len(data_dir))
    # for classes in labels_virus_bubble:
    #
    #     print()

    images_dir = os.listdir(data_dir_path)
    random.shuffle(images_dir)
    # random.Random(4).shuffle(images_dir)
    # images_dir = images_dir[0:40]

    # images_dir = images_dir[int(len(images_dir) * 0): int(len(images_dir) * .9)]
    # images_dir = images_dir[int(len(images_dir) * 0.7): int(len(images_dir) * 1)]
    # images_dir = images_dir[int(len(images_dir) * 0.7): int(len(images_dir) * 0.9)]
    # images_dir = images_dir[int(len(images_dir) * 0.9): int(len(images_dir) * 1)]
    # print(len(images_dir))

    for img_name in images_dir:
        # print(img_name,classes)
        # if ".jpg" in img_name:
        with open('../Data/Test/' + datasrt_name + '/COVID19_test_tile.txt', 'a') as the_file:
            # with open('../data/sd1/val.txt', 'a') as the_file:
            # the_file.write(data_dir_path+img_name+" "+img_name+'\n')
            the_file.write(
                data_dir_path + img_name + " " + "unlabelled" + '\n')

dataset_gen_txt('Test/mb1')
#gen_target('COVID19')

def remove_Spaces(data_path = "../Data/Test/mb1/Negative/"):
    for file in os.listdir(data_path):
       os.rename(data_path+file,data_path+file.replace(" ",""))
       #os.rename(data_path + file, data_path + file.replace(",", ""))
    # f = open("../Data/ZIKA/ZIKA_target_tile.txt","rt")
    # data = f.read()
    # data = data.replace('//','/')
    # f.close()
    # f = open("../Data/ZIKA/ZIKA_target_tile.txt", "wt")
    # data = f.write(data)
    # f.close()
# remove_Spaces()