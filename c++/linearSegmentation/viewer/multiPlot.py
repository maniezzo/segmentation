import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

if __name__ == "__main__":
   arrNames = [
      "AIC", "BIC", "Chi2",
      "MSE", "QRMSE", "Var",
      "R2", "RMSE", "SER"
   ]
   
   rows = []
   resFile = "../risultati.csv"
   with open(resFile, "r", encoding="utf-8") as f:
      header = next(f)  # skip header line
      for line in f:
         line = line.strip()
         if not line:
            continue
         
         parts = [p for p in line.split(",") if p != ""]  # remove empty fields from trailing commas
         
         name = parts[0]
         costFun = parts[1]
         tcpu = float(parts[2])  # or int(parts[2]) if always integer
         
         # remaining values (segment breakpoints)
         points = list(map(int, parts[3:]))
         
         rows.append({"name": name, "costFun": costFun, "tcpu": tcpu, "points": points})
   
   dfRes = pd.DataFrame(rows)
   print(dfRes.head())
   
   name = "M6/USDJPY"
   dfRes = dfRes[dfRes["name"] == name]
   
   path = "../../../data/"
   seriesFile = path + name + ".csv"
   dfPoints = pd.read_csv(seriesFile)
   
   # data series points
   y = dfPoints.iloc[1:, 1].to_numpy()
   x = np.arange(len(y))
   
   # Build one "lines" list per cost metric
   all_lines = []  # length = 9, each entry is a list of segments for that subplot
   
   for fcost in arrNames:
      df = pd.read_csv(path + name + "_cost" + fcost + "_segments.csv")
      lstOLS = df.values
      print(f"Cost {fcost} num. segments: {len(lstOLS)}")

      lines = []
      for j in range(len(lstOLS)):
         id = lstOLS[j][0]
         m  = lstOLS[j][3]
         q  = lstOLS[j][4]
         x1 = lstOLS[j][1]
         y1 = m*x1+q
         x2 = lstOLS[j][2]
         y2 = m*x2+q
         segm = [(x1,y1),(x2,y2)]
         lines.append(segm)
      #lc = mpl.collections.LineCollection(lines, linewidths=2, color = 'r', label = "OLS segments")
      all_lines.append(lines)

   fig,axes = plt.subplots(3,3,figsize=(10,10))
   axes = axes.flatten()
   
   for i, ax in enumerate(axes):
      ax.plot(dfPoints.iloc[1:, 1], marker='.', linewidth=0, color='b')
      
      # create a NEW LineCollection for this Axes using its own lines
      lc = mpl.collections.LineCollection(all_lines[i], linewidths=2, color="r")
      ax.add_collection(lc)
      
      ax.autoscale()
      ax.margins(0.1)
      ax.set_title(arrNames[i])
      ax.grid(True)
   
   plt.tight_layout()
   plt.savefig(f"{name.split('/')[-1]}_global.eps", bbox_inches='tight', format='eps')
   plt.show()
   print("fine")