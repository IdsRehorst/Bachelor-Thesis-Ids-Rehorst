import numpy as np
import matplotlib.pyplot as plt
import sys

coords = np.loadtxt(sys.argv[1], dtype=int)
plt.figure(figsize=(6,6))
plt.scatter(coords[:,1], coords[:,0], s=1)
plt.gca().invert_yaxis()
plt.title("Non‑zeros after RACE permutation")
plt.xlabel("column");  plt.ylabel("row")
plt.tight_layout()
plt.show()
