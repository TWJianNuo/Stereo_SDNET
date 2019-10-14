import os
import shutil
import errno
from cityscapesscripts.preparation.createTrainIdInstanceImgs import createTrainIdInstanceImgs
from cityscapesscripts.preparation.createTrainIdLabelImgs import createTrainIdLabelImgs
# import cityscapesscripts.preparation.createTrainIdInstanceImgs
# from cityscapesscripts.preparation.createTrainIdInstanceImgs import createTrainIdInstanceImgs
# from cityscapesscripts.preparation import createTrainIdLabelImgs
# from cityscapesscripts.preparation import createTrainIdInstanceImgs
class PreapareCityscape():
    def __init__(self, cts_path, fold_appen):
        # create a new folder, copy files, change copied files
        self.cts_path = cts_path
        self.anno_level = "gtFine+gtCoarse"
        self.fold_appen = fold_appen
        # self.labels = self.get_defined_traininglabel()
        self.copy_folder()
        self.process_label()

    def copy_folder(self):
        folders = self.anno_level.split('+')
        for folder in folders:
            src = os.path.join(self.cts_path, folder)
            dst = os.path.join(self.cts_path, folder + self.fold_appen)
            try:
                shutil.copytree(src, dst)
                print("Finish copying \"%s\" to \"%s\"" % (src, dst))
            except OSError as e:
                # If the error was caused because the source wasn't a directory
                if e.errno == errno.ENOTDIR:
                    shutil.copy(src, dst)
                elif e.errno == errno.EEXIST:
                    print("Detect folder already exist, skip copying")
                else:
                    print('Directory not copied. Error: %s' % e)
    def process_label(self):
        createTrainIdInstanceImgs(self.cts_path, "gtFine"+self.fold_appen, "gtCoarse"+self.fold_appen)
        print("")
        createTrainIdLabelImgs(self.cts_path, "gtFine"+self.fold_appen, "gtCoarse"+self.fold_appen)