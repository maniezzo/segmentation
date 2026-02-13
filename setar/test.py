import os
import sys

# 1. Define your R Home and Binaries
r_home = r'C:\Program Files\R\R-4.5.2'
r_bin = os.path.join(r_home, 'bin', 'x64')

# 2. Tell the OS where to find the R DLLs (Crucial for R 4.2+)
if sys.platform == 'win32':
   os.environ['R_HOME'] = r_home
   os.add_dll_directory(r_bin)

# 3. Now perform the imports
try:
   import rpy2.robjects as ro
   from rpy2.robjects.packages import importr
   from rpy2.robjects import pandas2ri
   print("✅ Connection to R 4.5.2 established successfully!")
except ImportError as e:
   print(f"❌ Still failing. Error: {e}")
   # If it fails here, your 'rpy2' folder in site-packages is likely missing __init__.py