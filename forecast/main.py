import os,pandas as pd
import mainSegments,mainOpt,util

if __name__ == '__main__':
   m          = -1 # stagionalità
   validation = -1 # numero dati da prevedere
   lstResults  = [] # tabella risultati finali
   with open("test_instances.txt", "r") as f:
      for line in f:
         filename = line.strip()
         
         # Stop at first empty row
         if not filename:
            break
         
         print(filename)
   
         nameCheck,categ,dfpoints,m,validation,min,max = util.read_series(filename)
         mainSegments.go_segment(nameCheck, dfpoints, m, validation)  # genera i segmenti se non ci sono già
         
         dfModels = pd.read_csv("data/" + nameCheck + f"models_{validation}.csv",usecols=["t1","t2", "AR1","HW","theta","RF"],)
         # Read binary
         #dfModels = pd.read_pickle("data/" + nameCheck + f"models_{validation}.pkl").loc[:, ["t1","t2", "AR1","HW","theta","RF"]]
         
         lstResults = mainOpt.go_opt(nameCheck, categ,  dfModels, dfpoints, m, validation, lstResults)
   dfResults = pd.DataFrame(lstResults)
   dfResults.to_csv("data/" + f"results_{validation}.csv", index=False)

   print("fine: >>>>>>>>>>>>>>> LEGGI note.txt <<<<<<<<<<<<<<<<")