import numpy as np, pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
   filenum = 0

   if(filenum==1 or filenum==0):
      df = pd.read_csv("..//MarMenorIP_OutputDiag.csv", skiprows=[0,2,3])
      for i in [3,4,11,15]:
         fig = plt.figure(figsize=(9,6))
         data = df.iloc[:,i]
         plt.plot(data)
         plt.title(f"{i} - {df.columns[i]}")
         plt.show()
         print(df.columns[i])
         data.to_csv(f"..//{df.columns[i]}.csv")

   if(filenum==2 or filenum==0):
      df = pd.read_csv("..//MarMenorIP_OutputDaily.csv", skiprows=[0,2,3])
      for i in [6,12,18]:
         fig = plt.figure(figsize=(9,6))
         data = df.iloc[:,i]
         plt.plot(data)
         plt.title(f"{i} - {df.columns[i]}")
         plt.show()
         print(df.columns[i])
         data.to_csv(f"..//{df.columns[i]}.csv")

   if(filenum==3 or filenum==0):
      df = pd.read_csv("..//MarMenorIP_Output60min.csv", skiprows=[0,2,3])
      for i in [4,5,11,12]:
         fig = plt.figure(figsize=(9,6))
         data = df.iloc[100:,i]
         plt.plot(data)
         plt.title(f"{i} - {df.columns[i]}")
         plt.show()
         print(df.columns[i])
         data.to_csv(f"..//{df.columns[i]}.csv")

   pass