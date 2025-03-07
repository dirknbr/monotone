
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt

def is_monotonic(xmin, xmax, f, bins=20, increase=True):
  x = np.linspace(xmin, xmax, bins)
  if increase:
    chg = sum([f(x[i]) >= f(x[i - 1]) for i in range(1, len(x))])
  else:
    chg = sum([f(x[i]) <= f(x[i - 1]) for i in range(1, len(x))])
  return chg == len(x) - 1

def find_model(x, y, maxdegree=10, method='poly', bins=20, increase=True, k=3):
  """
    Args:
      x: univariate array
      y: univariate array
      maxdegree: integer for poly search
      method: poly or spline
      bins: number of bins for monotonic check
      increase: boolean, has x positive effect
      k: degree for spline
  """

  assert method in ['poly', 'spline']
  xmin, xmax = x.min(), x.max()
  if method == 'poly':
    best = {'corr': -1}
    for deg in range(1, maxdegree + 1):
      params = np.polyfit(x, y, deg)
      f = np.poly1d(params)
      corr = np.corrcoef(y, f(x))[0, 1]
      is_mon = is_monotonic(xmin, xmax, f, bins=bins, increase=increase)
      if corr > best['corr'] and is_mon:
        best = {'degree': deg, 'f': f, 'corr': corr}
  else:
    # sort (x, y) by x
    x2 = x[np.argsort(x)]
    y2 = y[np.argsort(x)]
    f = interpolate.UnivariateSpline(x2, y2, k=3)
    corr = np.corrcoef(y, f(x))[0, 1]
    best = {'degree': k, 'f': f, 'corr': corr, 'is_monotonic': 
      is_monotonic(xmin, xmax, f, bins=bins, increase=increase)}
  return best

def plot(x, y, f):
  plt.scatter(x, y)
  plt.plot(x, f(x), color='green')
