import csv
import ast
from scipy import interpolate
import matplotlib
import matplotlib.pyplot as plt
import os

matplotlib.use('Agg') 

####
# This program converts the doodling lines into images (.png).
####

TEST = False
SAMPLE_NUM = 3001
CSV_DIR = '../Warehouse/dataset/train_csv/'
SAVE_DIR = '../Warehouse/dataset/train/'
class_list = {'airplane': 0, 'bee': 1, 'bicycle': 2, 'bird': 3, 'butterfly': 4, 'cake': 5,
    'camera': 6, 'cat': 7, 'chair': 8, 'clock': 9, 'computer': 10,
    'diamond': 11, 'door': 12, 'ear': 13, 'guitar': 14, 'hamburger': 15,
    'hammer': 16, 'hand': 17, 'hat': 18, 'ladder': 19, 'leaf': 20,
    'lion': 21, 'pencil': 22, 'rabbit': 23, 'scissors': 24, 'shoe': 25,
    'star': 26, 'sword': 27, 'The Eiffel Tower': 28, 'tree': 29}

def convert_csv_to_png(file_name, save_dir):
    datafile = open(file_name, 'r')
    list_file = open(save_dir + "train_list.txt", "a+")

    datareader = csv.reader(datafile, delimiter=',')
    data = []
    idx = 0
    for data in datareader: # for each img...
        idx = idx + 1
        if(idx == 1): continue
        if(idx == SAMPLE_NUM): break

        num = data[0]
        stroke = data[1]
        label = data[2]

        if not os.path.exists(save_dir + label):
            os.makedirs(save_dir + label)
        
        stroke_array = ast.literal_eval(stroke)
        num_of_strokes = len(stroke_array)

        for i in range(num_of_strokes):
            x = stroke_array[i][0]
            y = stroke_array[i][1]
            f = interpolate.interp1d(x, y, kind="slinear")
            plt.plot(x, y, 'k')

        ax = plt.gca()
        ax.xaxis.set_ticks_position('top')
        ax.invert_yaxis()
        plt.axis('off')
        if (TEST):
            file_name = num + ".png"
        else:
            file_name = label + "/" + num + ".png"
        plt.savefig(save_dir + file_name)
        plt.close()

        list_file.write(file_name + "\n")

        print("-----------")
        print("Class: ", label)
        print("Num: ", num)
        
    list_file.close()

if __name__ == '__main__':
    for c, v in list(class_list.items()):
        convert_csv_to_png(CSV_DIR + c + ".csv", SAVE_DIR)
