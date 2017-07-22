import logRegres
from numpy import *

dataArr,labelMat = logRegres.loadDataSet()
weights = logRegres.stocGrandAscent1(array(dataArr),labelMat)
#weights =logRegres.gradAscent(dataArr,labelMat)
#mat转为array
#对一维的数组转制不起作用，改为mat类型矩阵好点
logRegres.plotBestFit(mat(weights).transpose())

#logRegres.multiTest()