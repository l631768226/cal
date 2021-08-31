import pyreadr
from collections import OrderedDict

result = pyreadr.read_r('E:/1.rds')
print(result.values())