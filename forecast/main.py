import os,pandas as pd
import mainSegments,mainOpt,util

if __name__ == '__main__':
   
   with open("test_instances.txt", "r") as f:
      for line in f:
         filename = line.strip()
         
         # Stop at first empty row
         if not filename:
            break
         
         print(filename)
   
         nameCheck,dfpoints = util.read_series(filename)
         isTheta = True
         mainSegments.go_segment(nameCheck, dfpoints, isTheta)  # genera i segmenti se non ci sono già
         dfModels = pd.read_csv("data/" + nameCheck + "models.csv", usecols=['0', '1', '2', '3', '4'])
         mainOpt.go_opt(nameCheck, dfModels, dfpoints)

   print("fine: >>>>>>>>>>>>>>> LEGGI note.txt <<<<<<<<<<<<<<<<")