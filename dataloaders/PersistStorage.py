import sys , os
import json

DEBUG = True

class PersistStorage(object): # ./Datatools/Configs/OS_linux
    def __init__(self, anchor=None):
        self.dirThisFolder = os.path.dirname(__file__)
        self.dirCWD= os.getcwd()
        if anchor is None:
            self.anchor = os.path.join(self.dirCWD, "data")
        else:
            self.anchor= anchor
        self.anchor= os.path.abspath(self.anchor)

    def getName(self, shortName=None):
        if shortName is None:
            return self.anchor
        assert isinstance(shortName, list) or isinstance(shortName, str)
        if isinstance(shortName, list):
            _shortName="."
            for folderName in shortName:
                _shortName = os.path.join(_shortName, folderName)
            return self.getName(_shortName)

        return os.path.abspath(os.path.join(self.anchor, shortName))

    def getJSON(self, filePath): # StorageDir
        if not filePath.split(".")[-1]=="json":
            filePath+=".json"
        with open(self.getName(filePath), "r") as f :
            jsonFile = json.load(f)
            return jsonFile

    def getTXT(self, filePath): # StorageDir
        if not filePath.split(".")[-1]=="txt":
            filePath+=".txt"
        with open(self.getName(filePath), "r") as f :
            txtFile = f.readlines()
            return txtFile
    def writeTXT(self, filePath, str):
        if not filePath.split(".")[-1]=="txt":
            filePath+=".txt"
        with open(self.getName(filePath), "a+") as f :
            f.write(str)
            # f.write('\r\n')
            f.close()
            # return txtFile



    def getFolderContent(self,folderDir='.'):
        fullDirPath=self.getName(folderDir)
        allFiles=os.listdir(fullDirPath)
        files,folders=list(),list()
        for x in allFiles:
            if os.path.isfile(os.path.join(fullDirPath,x)):
                files.append(os.path.join(fullDirPath,x))
            elif os.path.isdir(os.path.join(fullDirPath,x)):
                folders.append(os.path.join(fullDirPath,x))
        return{"all":allFiles,"folder":folders,"files":files}










