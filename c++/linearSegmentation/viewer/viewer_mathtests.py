import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

def plotSegments(x,y,baseDir,name,costFun):
   dfSol = pd.read_csv(f"..//..//..//data//{baseDir}//{name}_cost{costFun}_segments.csv");
   lstOLS = dfSol.values
   costf = dfSol.columns[-1]
   
   print(f"num. segments: {len(lstOLS)}")
   lines = []
   for j in range(len(lstOLS)):
      id = lstOLS[j][0]
      m = lstOLS[j][3]
      q = lstOLS[j][4]
      x1 = lstOLS[j][1]
      y1 = m * x1 + q
      x2 = lstOLS[j][2]
      y2 = m * x2 + q
      segm = [(x1, y1), (x2, y2)]
      lines.append(segm)
   
   lc = mpl.collections.LineCollection(lines, linewidths=2, color='r', label="OLS segments")
   
   fig, ax = plt.subplots()
   ax.plot(x, y, 'o', label='Original data', markersize=3,linestyle='-',linewidth=0.5)
   ax.add_collection(lc)
   ax.autoscale()
   ax.margins(0.1)
   plt.legend()
   plt.title(f"{name} - {costf}")
   if (fSaveFig):
      plt.savefig(f"{name}-{costf}.eps", bbox_inches='tight', format='eps')
   plt.show()
   print(f"{name} - {costf}")
   return

if __name__ == "__main__":
   #plt.ion()  # Turn on interactive mode
   plt.ioff()
   mpl.use('Qt5Agg')
   print("backend: "+mpl.get_backend())
   baseDir = "M3"
   if(baseDir == "mathtests"):
      ds = pd.read_csv(f"..//..//..//data//mathtests//M3_4_sample.csv")
      idFile = 54  # index in M3_4_sample.csv
      name = ds.iloc[idFile,1]
      y = ds.iloc[idFile,4:].dropna().values
   elif(baseDir == "M3"):
      df = pd.read_csv(f"..//..//..//data//M3//M3month.csv")
      name = "N2798"
      ds = df[df.iloc[:, 0] == name]
      y = ds.iloc[0,6:].dropna().values
   x = np.arange(len(y))
   fSaveFig = True
   for costFun in ["QRMSE","AIC"]:
      plotSegments(x, y, baseDir, name, costFun)
   print("fine")