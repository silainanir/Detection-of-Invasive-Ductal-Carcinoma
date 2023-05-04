from PIL import Image, ImageStat
import datetime
import os


class RGB:

    def __init__(self, imagePathList):
        self.imagePathList = imagePathList

    """
    Image'nin 50x50 ->> 2500 pixelinin rgb değerlerini buluyor. 
    """

    def get_rgb_code(self, image_path):
        im = Image.open(image_path)
        pix = im.load()
        rgbOfImageList = list()
        stat = ImageStat.Stat(im)

        """
        her image için rgb listi oluşturuyor.
        listin 2500 elemanı var.
        (R, G, B) * 2500
        """
        red = 0
        green = 0
        blue = 0
        for x in range(im.width):
            for y in range(im.height):
                rgbOfImageList.append(pix[x, y])
                red += pix[x, y][0]
                green += pix[x, y][1]
                blue += pix[x, y][2]
        red /= im.width * im.height
        green /= im.width * im.height
        blue /= im.width * im.height

        # rgbOfImageList.append(im.getextrema())
        # rgbOfImageList.append(im.entropy())
        # rgbOfImageList.append(red)
        # rgbOfImageList.append(green)
        # rgbOfImageList.append(blue)
        # rgbOfImageList.append(stat.stddev)

        return rgbOfImageList

    def get_image_paths(self, root):
        """
        Tries for a patient and obtains path for each image regardless of class value
        :param root: The directory of the current patient
        :return: Two lists which include image paths for class 0 and class 1 separately
        """
        class0_rgb = []
        class1_rgb = []
        for root, dir, files in os.walk(root, topdown=True):
            if root[-1] == '0':
                class0_rgb = files
            elif root[-1] == '1':
                class1_rgb = files

        return class0_rgb, class1_rgb



    def get_RGB_color_codes(self):
        print(len(self.imagePathList))
        retList = list()
        # image pathlerini tek tek rgb bulan fonksiyona yolluyor.
        # gelen her return değeri liste ekliyor

        """
        path--> list 
        
        i-->path
        """
        # start = datetime.datetime.now()
        # print("start:", start)
        # counter = 0
        for i in self.imagePathList:
            # if counter == 10000:
            #     break
            # if counter % 1000:
            #     print("now:", datetime.datetime.now())
            #
            # counter += 1
            # print(i)
            # retList.append(list)
            # class, id, x, y, rgbList
            # Bu pathi burdan almam lazım.
            forDataBase = i.replace("Dataset/", "") \
                .replace("_class1.png", "").replace("_class0.png", "") \
                .replace("_idx5", "").replace("x", "").replace("y", "").split("/")

            forDataBase = forDataBase + forDataBase[2].split("_")

            del forDataBase[2]
            del forDataBase[0]
            forDataBase = [int(i) for i in forDataBase]
            forDataBase[1] = str(forDataBase[1])

            imageRGB_list = self.get_rgb_code(i)
            rgbMinMax = imageRGB_list[-6]
            rgbMinMax = str(rgbMinMax)

            imageEntropy = imageRGB_list[-5]
            R_Average = imageRGB_list[-4]
            G_Average = imageRGB_list[-3]
            B_Average = imageRGB_list[-2]

            image_std = imageRGB_list[-1]

            rgbMinMax = rgbMinMax.replace("(", "").replace(")", "").split(",")
            rgbMinMax = [int(i) for i in rgbMinMax]
            # print(rgbMinMax)

            # forDataBase += rgbMinMax

            forDataBase.append(rgbMinMax[0])
            forDataBase.append(rgbMinMax[1])
            forDataBase.append(R_Average)
            forDataBase.append(image_std[0])

            forDataBase.append(rgbMinMax[2])
            forDataBase.append(rgbMinMax[3])
            forDataBase.append(G_Average)
            forDataBase.append(image_std[1])

            forDataBase.append(rgbMinMax[4])
            forDataBase.append(rgbMinMax[5])
            forDataBase.append(B_Average)
            forDataBase.append(image_std[2])

            forDataBase.append(float(imageEntropy))

            for i in range(6):
                del imageRGB_list[-1]
            forDataBase.append(imageRGB_list)

            # print(forDataBase)
            retList.append(forDataBase)

        # end = datetime.datetime.now()
        # print("Time elapsed for", counter, " pics: ", (end - start))
        return retList
