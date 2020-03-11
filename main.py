"""cv_hw1.py: Starter file to run howework 2"""

#Example Usage: ./cv_hw1 -i image -k clusters -m grey
#Example Usage: python cv_hw1 -i image -k clusters -m rgb


__author__      = "Pranav Mantini"
__email__ = "pmantini@uh.edu"
__version__ = "1.0.0"

import cv2
import sys
from Segmentation.KMeans import KmeansSegmentation
from datetime import datetime
##%%
# display_image("image","fruits.jpg")
# # %%
# def IMGstatistics(image_path, color, output_path):
#     import matplotlib.pyplot as plt
#     if color == 'GRAY':
#         hist = np.zeros([256], dtype = np.int32)
#         image = cv2.imread(image_path, 0)
#         height, width = image.shape[0], image.shape[1]
#         if image is None:
#             print("image reading failure, please check image's path!")
#             exit()
#         for row in range(height):
#             for col in range(width):
#                 pv = image[row, col]
#                 hist[pv] +=1
#         plt.figure()
#         plt.title(image_path + " Gray Histogram")
#         plt.xlabel("intensity")
#         plt.ylabel("number of pixels")
#         # plt.hist(hist, bins=8000, density = 'TRUE', facecolor = 'green', alpha = 0.5)
#         plt.plot(hist, color="r")
#         plt.xlim([0,256])
#         plt.savefig(output_path + "GRAY_statictics_" + image_path)
#         plt.show()
#     elif color == 'RGB':
#         image = cv2.imread(image_path)
#         height, width = image.shape[0], image.shape[1]
#         if image is None:
#             print("image reading failure, please check image's path!")
#             exit()
#         histb = np.zeros([256], dtype = np.int32)
#         histg = np.zeros([256], dtype = np.int32)
#         histr = np.zeros([256], dtype = np.int32)
#         for row in range(height):
#             for col in range(width):
#                 pvb = image[row, col, 0]
#                 histb[pvb] +=1
#                 pvg = image[row, col, 1]
#                 histg[pvg] += 1
#                 pvr = image[row, col, 2]
#                 histr[pvr] += 1
#         plt.figure()
#         plt.title(image_path + " RGB Histogram")
#         plt.xlabel("intensity")
#         plt.ylabel("number of pixels")
#         plt.plot(histb, color = 'b')
#         plt.plot(histg, color = 'g')
#         plt.plot(histr, color = 'r')
#         plt.xlim([0,256])
#         plt.savefig(output_path + "RGB_statictics_" + image_path)
#         plt.show()
#     else :
#         print("input type error!")
#         pass
# # %%
# image_path = 'circles.jpg'
# color = 'RGB'
# output_path = '/home/cougarnet.uh.edu/yzi2/Pictures/'
# IMGstatistics(image_path, color, output_path)
#
# # %%
def main():
    """ The main funtion that parses input arguments, calls the approrpiate
     kmeans method and writes the output image"""

    #Parse input arguments
    from argparse import ArgumentParser

    parser = ArgumentParser()

    parser.add_argument("-i", "--image", dest="image",
                        help="specify the name of the image", metavar="IMAGE")
    parser.add_argument("-k", "--clusters", dest="clusters",
                        help="Specify the number of clusters (k)", metavar="CLUSTERS")
    parser.add_argument("-m", "--model", dest="model",
                        help="Specify the model rgb, grey", metavar="COLOR")

    args = parser.parse_args()

    #Load image
    if args.image is None:
        print("Please specify the name of image")
        print("use the -h option to see usage information")
        sys.exit(2)
    else:
        image_name = args.image.split(".")[0]
        input_image = cv2.imread(args.image)


    if args.clusters is None:
        print("Number of clusters not specified using 2")
        print("use the -h option to see usage information")
        clusters = 2
    else:
        clusters = int(args.clusters)

    # Check resize scale parametes
    if args.model is None:
        print("Model not specified using default (grey)")
        print("use the -h option to see usage information")
        model = 'grey'
    elif args.model not in ['rgb', 'grey']:
        print("Unknown color model, using default (grey)")
        print("use the -h option to see usage information")
        model = 'grey'
    else:
        model = args.model

    Segementation_object = KmeansSegmentation()

    output = None
    if model == 'grey':
        input_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)
        output = Segementation_object.segmentation_grey(input_image, clusters, image_name)
    else:
        output = Segementation_object.segmentation_rgb(input_image, clusters, image_name)