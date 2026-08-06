import os,pandas as pd
import mainSegments,mainOpt,util

if __name__ == '__main__':
   
   m = 12
   validation = 6 # numero dati da prevedere
   with open("test_instances.txt", "r") as f:
      for line in f:
         filename = line.strip()
         
         # Stop at first empty row
         if not filename:
            break
         
         print(filename)
   
         nameCheck,dfpoints = util.read_series(filename)
         mainSegments.go_segment(nameCheck, dfpoints, m, validation)  # genera i segmenti se non ci sono già
         dfModels = pd.read_csv("data/" + nameCheck + "models.csv",usecols=["t1","t2","HW","theta","RF"],)
         mainOpt.go_opt(nameCheck, dfModels, dfpoints, m, validation)

   print("fine: >>>>>>>>>>>>>>> LEGGI note.txt <<<<<<<<<<<<<<<<")