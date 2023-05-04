import os


class FileOperations:

    def __init__(self, pathOfDataSet=""):
        self.pathOfDataSet = pathOfDataSet

    """
    Tüm imagelerin pathlerini list olarak return ediyor.
    """

    def read_images_from_set(self):
        pathList = list()
        for root, dirs, files in os.walk(self.pathOfDataSet):
            for file in files:
                # print(file)
                pathList.append(root.replace('\\', '/') + "/" + file)
        return pathList

    """
    İki farklı texte atabilirim.
    Ama her seferinde dosyayı açıp kapatmam lazım.
    Bunda direkt liste atıp listi bastırıyorum.
    """

    def read_patient_folders(self):
        """
        Reads folder names of patients
        :return: list which includes folder names of patients
        """
        for root, dirs, files in os.walk(self.pathOfDataSet):
            return dirs

    def write_2_text(self, to_be_WrittenList, filename, mode="a"):
        toFile = open("FileOperations/" + filename, mode)
        # for element in to_be_WrittenList:
        toFile.write(str(to_be_WrittenList) + "\n")
        toFile.close()
