import csv
import numpy as np
import pandas as pd
pd.set_option('max_rows', None)
pd.set_option('max_columns', None)

types={'id':int, 'y': int, 'x1':float, 'x2':float,'x3':float,'x4':float,'x5':int,'x6':int, 'x7':float, 'x8':float, 'x9':float, 'x10':float }
etypes={'id':int, 'x1':float, 'x2':float,'x3':float,'x4':float,'x5':int,'x6':int, 'x7':float, 'x8':float, 'x9':float, 'x10':float }
betyg = {'F':1, 'Fx':2, 'E':3,'D':4, 'C':5,'B':6, 'A':7}
TF = {True:1,False:0}
matesPLz = {'Atsuto':0, 'Bob': 1, 'Jörg':2}
xl={'x1':0, 'x2':1, 'x3':2, 'x4':3, 'x5':4, 'x6':5, 'x7':6, 'x8':7, 'x9':8, 'x10':9 }

'''
Most reused from the labs since we could use any library!
Ignore the horrible naming! I had a rough time.

'''



class trainData:

    def __init__(self):
        self.train = pd.read_csv('TrainOnMe.csv')
        self.all = self.train.columns.to_numpy()

        self.xLabels = self.train.columns[2:].to_numpy()
        self.Xs,self.Ys = self.fixFile()
        self.labels = np.unique(self.Ys)

    def fixFile(self):
        t = self.train
        threshold = 5
        tData = pd.DataFrame(columns=self.all)
        tData = tData.astype(types)
        t.fillna(0, inplace=True)




        for index,row in t.iterrows():
            a = pd.Series.isna(t.loc[index]).values.sum()
            if a == 0:
                tData.loc[index] = self.change(row)

            if a > 0 and a < threshold:
                tData.loc[index] = self.change(self.shift(row,a)[index])




        Ys = tData['y']
        Xs = tData[self.xLabels]
        return Xs,Ys.to_numpy()

    def getXvalue(self,x):
        return self.Xs[x].to_numpy()
    def getX(self):
        return self.Xs

    def change(self, row):
        columns = self.all
        temp = {}
        for o in types:
            try:

                if o == 'x5':
                    temp[o] = [int(TF[row[o]])]
                elif o == 'x6':
                    temp[o] = [int(betyg[row[o]])]
                elif o == 'id':
                    temp[o] = [int(row[o])]
                elif o == 'y':
                    temp[o] = [int(matesPLz[row[o]])]
                else:
                    temp[o] = [float(row[o])]

            except:
                temp[o] = int(0)

        out = pd.DataFrame(data=temp, columns=columns)
        out = out.astype(types)
        return out.loc[0]

    def shift(self,row,push):
        t = row.to_frame()
        a = t[0:2]
        b = t[2:].shift(periods=push)
        out = pd.concat([a,b],axis=0)
        return out

class evalSep:
    def __init__(self):
        self.evaluate = pd.read_csv('EvaluateOnMe.csv',delimiter=',')
        self.all = self.evaluate.columns.to_numpy()
        self.Xs= self.getValues()


    def getValues(self):
        eData = pd.DataFrame(columns=self.all)
        eData = eData.astype(etypes)

        xlabels = eData.columns[1:]
        for o in etypes:
            if o == 'x6':
                eData[o] = self.evaluate[o].apply(self.grading).astype(etypes[o])
            elif o == 'x5':
                eData[o] = self.evaluate[o].apply(self.tf).astype(etypes[o])

            else:
                eData[o] = self.evaluate[o].astype(etypes[o])

        X = eData[xlabels]


        return X


    def grading(self, a):
        return betyg[a]
    def tf(self,a):
        return TF[a]