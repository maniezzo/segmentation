import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import chow_test
'''
git clone https://github.com/jtloong/chow-test.git
cd chow-test
pip install .
remove chow-test directory
In your Python code you can import it as:
import chow_test
The function has four parameters, and be used to find either the f-value or p-value of your Chow test.
chow_test.f_value(y1, x1, y2, x2)
or
chow_test.p_value(y1, x1, y2, x2)
'''
if __name__ == "__main__":
   plt.ion()  # Turn on interactive mode
   #mpl.use('Qt5Agg')
   print(mpl.get_backend())
   dataFiles = ["test","BTC-USD","IBatt_Min","PTemp_C_Avg","PTemp_C_Max","V_in_chg_Avg","Vapor_Pressure_Avg","ThermTemp1_Avg","WTemp_C2_Avg","SDI_Temp_1m","WS_ms_Avg","WTemp_C1_Avg","Vapor_Pressure_Avg_2"]
   idFile = 4
   name = dataFiles[idFile]
   df = pd.read_csv(f"..//..//..//data//home//{name}.csv");
   y = df[name].values
   x = np.arange(len(y))

   dfSol = pd.read_csv(f"..//..//..//data//home//{name}_segments.csv")
   lstOLS = dfSol.values
   costf = dfSol.columns[-1]

   print(f"num. segments: {len(lstOLS)}")
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

   # chow test, significance of break betwween segments s1 and s2
   x10 = int(lstOLS[0][1])
   x11 = int(lstOLS[0][2])
   x20 = int(lstOLS[1][1])
   x21 = int(lstOLS[1][2])
   xs1 = x[x10:x11]
   ys1 = y[x10:x11]
   xs2 = x[x20:x21]
   ys2 = y[x20:x21]
   pchow = chow_test.p_value(ys1, xs1, ys2, xs2)
   print(f"Prob. chow test: {pchow}")

   lc = mpl.collections.LineCollection(lines, linewidths=2, color = 'r', label = "OLS segments")

   fig, ax = plt.subplots()
   ax.plot(x, y, 'o', label='Original data', markersize=3)
   ax.add_collection(lc)
   ax.autoscale()
   ax.margins(0.1)
   plt.legend()
   plt.title(f"{idFile}) {name} - {costf}")
   plt.savefig(f"{idFile}) {name} - {costf}.eps", bbox_inches='tight', format='eps')
   plt.show()
   print(f"{idFile}) {name} - {costf}")