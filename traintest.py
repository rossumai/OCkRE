from ockre import OCkRE
from synthset import CropImageIterator
import fakestrings

ocr = OCkRE()
ocr.train("test_run",0,1000,epochlen=2048,vallen=2048)