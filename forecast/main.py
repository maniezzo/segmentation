import pandas as pd
import mainSegments,mainOpt,util

if __name__ == '__main__':
   name="N1930" #"N1679" "N1402"
   nameCheck,dfpoints = util.read_series(name)
   dfModels = pd.read_csv("data/" + name + "models.csv", usecols=['0', '1', '2', '3', '4'])
   isTheta = True
   mainOpt.go_opt(name, dfModels, dfpoints, isTheta)

   print("fine: >>>>>>>>>>>>>>> LEGGI note.txt <<<<<<<<<<<<<<<<")