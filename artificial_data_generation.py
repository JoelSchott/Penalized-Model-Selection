import numpy as np
import pandas as pd

N = 100
np.random.seed(0)

x1 = np.random.rand(N)
x2 = (x1 * 2) + np.random.randn(N)
x3 = np.random.rand(N) * 2
y = x1 + x3 + np.random.rand(N)

data = pd.DataFrame({'y': y, 'x1': x1, 'x2': x2, 'x3': x3})
data.to_csv("artificial_data.csv", index=False)



