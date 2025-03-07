
import unittest
from model import *

class ModelTests(unittest.TestCase):

  def test_monotonic(self):
  	f1 = lambda x: x
  	response = is_monotonic(0, 10, f1)
  	self.assertTrue(response)

  	f2 = lambda x: -x
  	response = is_monotonic(0, 10, f2, increase=False)
  	self.assertTrue(response)

  def test_poly_increase(self):
    x = np.arange(20)
    y = 1 + x ** .5
    best = find_model(x, y)
    self.assertGreater(best['corr'], .99)
    self.assertEqual(best['degree'], 10)

  def test_poly_decrease(self):
    x = np.arange(20)
    y = 1 - x ** .5
    best = find_model(x, y, increase=False)
    self.assertGreater(best['corr'], .99)
    self.assertEqual(best['degree'], 10)

  def test_spline_and_plot(self):
    x = np.arange(20)
    y = 1 + x ** .5
    best = find_model(x, y, method='spline')
    plot(x, y, best['f'])
    plt.savefig('temp.png') # for the readme
    self.assertGreater(best['corr'], .99)
    self.assertTrue(best['is_monotonic'])

if __name__ == '__main__':
  unittest.main()