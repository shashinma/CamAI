import Modules.DataUpdater as DataUpdater
import Modules.VideoCapture as VideoCapture

def Startup():
    DataUpdater.DataUpdate()
    VideoCapture.VCapt()

if __name__ == "__main__":
    Startup()