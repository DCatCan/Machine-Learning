
import numpy as np
import dataSeparator as ds
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import tree



matesPLz = {'Atsuto':0, 'Bob': 1, 'Jörg':2}
lastshit = {0:'Atsuto', 1:'Bob', 2:'Jörg'}
#xl={'x1':0, 'x2':1, 'x3':2, 'x4':3, 'x5':4, 'x6':5, 'x7':6, 'x8':7, 'x9':8, 'x10':9 }

'''
Most reused from the labs since we could use any library!
Ignore the horrible naming! I had a rough time.

Time to study some more!
'''


e = ds.evalSep()
eX = e.getValues()
a = ds.trainData()
y = a.Ys
y = y.astype(int)
X = np.asarray(a.Xs)


def computePrior(labels, W=None):
    Npts = labels.shape[0]
    if W is None:
        W = np.ones((Npts,1))/Npts
    else:
        assert(W.shape[0] == Npts)
    classes = np.unique(labels)
    Nclasses = np.size(classes)


    prior = np.zeros((Nclasses,1))

    # TODO: compute the values of prior for each class!
    # ==========================
     #To check that the probabilities are summed up to 1

    for idx,c in enumerate(classes):
        idx = labels == c
        idx = np.where(labels==c)[0]
        prior[c] = np.sum(W[idx])

    #print(np.sum(prior))

    # ==========================

    return prior

def mlParams(X, labels, W=None):
    assert(X.shape[0]==labels.shape[0])
    Npts,Ndims = np.shape(X)
    classes = np.unique(labels)
    Nclasses = np.size(classes)
    if W is None:
        W = np.ones((Npts,1))/float(Npts)

    mu = np.zeros((Nclasses,Ndims))
    sigma = np.zeros((Nclasses,Ndims,Ndims))
    # TODO: fill in the code to compute mu and sigma!
    # ==========================
    #calculate mean for each class
    for idx,c in enumerate(classes):
        idx = np.where(labels==c)[0]
        xlc = X[idx]
        weight = np.sum(W[idx])

        mu[c] += np.sum(xlc*W[idx],axis=0)/weight

        for d in range(Ndims):
            for k in idx:
                sigma[c][d][d]+= W[k]*(X[k][d]-mu[c][d])**2
            sigma[c][d][d] = sigma[c][d][d]/weight

    # ==========================
    return mu, sigma


def classifyBayes(X, prior, mu, sigma):

    Npts = X.shape[0]
    Nclasses,Ndims = np.shape(mu)
    logProb = np.zeros((Nclasses, Npts))
    # TODO: fill in the code to compute the log posterior logProb!
    # =========================


    for k in range(Nclasses):
        newX = X - mu[k]  # Matrix  C x d with diffs between x - µ
        for i in range(Npts):
            logProb[k][i] = - np.log(np.linalg.det(sigma[k])) / 2 - np.inner(newX[i] / np.diag(sigma[k]),newX[i]) / 2 + np.log(prior[k])


    # ==========================

    # one possible way of finding max a-posteriori once
    # you have computed the log posterior
    h = np.argmax(logProb,axis=0)
    return h


class BayesClassifier(object):
    def __init__(self):
        self.trained = False

    def trainClassifier(self, X, labels, W=None):
        rtn = BayesClassifier()
        rtn.prior = computePrior(labels, W)
        rtn.mu, rtn.sigma = mlParams(X, labels, W)
        rtn.trained = True
        return rtn

    def classify(self, X):
        return classifyBayes(X, self.prior, self.mu, self.sigma)


#testClassifier(BayesClassifier(), X, y)


def trainBoost(base_classifier, X, labels, T=10):
    # these will come in handy later on
    Npts, Ndims = np.shape(X)

    classifiers = []  # append new classifiers to this list
    alphas = []  # append the vote weight of the classifiers to this list

    # The weights for the first iteration
    wCur = np.ones((Npts, 1)) / float(Npts)

    for i_iter in range(0, T):
        # a new classifier can be trained like this, given the current weights
        classifiers.append(base_classifier.trainClassifier(X, labels, wCur))

        # do classification for each point
        vote = classifiers[-1].classify(X)

        # TODO: Fill in the rest, construct the alphas etc.
        # ==========================
        '''
        step0: wCur = Initialize all weights
        step1: train weak learner using distribution w^t

        vote[i_iter] <= h^t
        epsilon over i_iter
        hyp if h(X[i])^t == labels[i] then 1 else 0
        '''
        # step2:
        hyp = np.ones((Npts, 1))
        for i in range(Npts):
            if vote[i] != labels[i]:
                hyp[i][:] = 0

        # step3: choose alpha

        err = np.sum(wCur * (1 - hyp))
        alpha = 0.5 * (np.log(1 - err) - np.log(err))

        alphas.append(alpha)  # you will need to append the new alpha

        newW = np.zeros((Npts, 1))
        # step4:  update w
        for i in range(Npts):
            if vote[i] == labels[i]:
                wCur[i][:] = wCur[i][:] * np.exp(-alpha)
            else:
                wCur[i][:] = wCur[i][:] * np.exp(alpha)

        Z = np.sum(wCur)
        wCur /= Z

        # ==========================

    return classifiers, alphas

def classifyBoost(X, classifiers, alphas, Nclasses):
    Npts = X.shape[0]
    Ncomps = len(classifiers)


    # if we only have one classifier, we may just classify directly
    if Ncomps == 1:
        return classifiers[0].classify(X)
    else:
        votes = np.zeros((Npts,Nclasses))

        # TODO: implement classificiation when we have trained several classifiers!
        # here we can do it by filling in the votes vector with weighted votes
        # ==========================

        for i in range(Ncomps):
            h = classifiers[i].classify(X)
            for k in range(Npts):
                #adding the alpha to the hypothesis for all the classes to later choose the max
                votes[k, h[k]] += alphas[i]

        # ==========================

        # one way to compute yPred after accumulating the votes
        return np.argmax(votes,axis=1)


class BoostClassifier(object):
    def __init__(self, base_classifier, T=10):
        self.base_classifier = base_classifier
        self.T = T
        self.trained = False

    def trainClassifier(self, X, labels):
        rtn = BoostClassifier(self.base_classifier, self.T)
        rtn.nbr_classes = np.size(np.unique(labels))
        rtn.classifiers, rtn.alphas = trainBoost(self.base_classifier, X, labels, self.T)
        rtn.trained = True
        return rtn

    def classify(self, X):
        return classifyBoost(X, self.classifiers, self.alphas, self.nbr_classes)

def testClassifier(classifier, X, y, dim=0, test=0.3, ntrials=100):

    pcadim =0

    means = np.zeros(ntrials,)

    for trial in range(ntrials):

        xTr,xTe,yTr,yTe= train_test_split(X,y,test_size=test,random_state=trial)


        # Train
        trained_classifier = classifier.trainClassifier(xTr, yTr)
        # Predict
        yPr = trained_classifier.classify(xTe)

        # Compute classification error
        if trial % 10 == 0:
            print("Trial:",trial,"Accuracy","%.3g" % (100*np.mean((yPr==yTe).astype(float))) )

        means[trial] = 100*np.mean((yPr==yTe).astype(float))

    print("Final mean classification accuracy ", "%.3g" % (np.mean(means)), "with standard deviation", "%.3g" % (np.std(means)))
    return trained_classifier.classify(eX)

class DecisionTreeClassifier(object):
    def __init__(self):
        self.trained = False

    def trainClassifier(self, Xtr, yTr, W=None):
        rtn = DecisionTreeClassifier()
        rtn.classifier = tree.DecisionTreeClassifier(max_depth=Xtr.shape[1]/2+1)
        if W is None:
            rtn.classifier.fit(Xtr, yTr)
        else:
            #flatten the dimensions to get a (,n) array
            rtn.classifier.fit(Xtr, yTr, sample_weight=W.flatten())
        return rtn

    def classify(self, X):
        return self.classifier.predict(X)


def testClassifier(classifier, X, y, test=0.2, n=100):


    means = np.zeros(n,)

    for trial in range(n):

        xTr,xTe,yTr,yTe= train_test_split(X,y,test_size=test,random_state=trial)

        # Train
        trained_classifier = classifier.trainClassifier(xTr, yTr)
        # Predict
        yPr = trained_classifier.classify(xTe)

        # Compute classification error
        if trial % 10 == 0:
            print("Trial:",trial,"Accuracy","%.3g" % (100*np.mean((yPr==yTe).astype(float))) )

        means[trial] = 100*np.mean((yPr==yTe).astype(float))

    print("Final mean classification accuracy ", "%.3g" % (np.mean(means)), "with standard deviation", "%.3g" % (np.std(means)))
    return trained_classifier.classify(eX)

#testClassifier(BoostClassifier(BayesClassifier(), T=10), X, y)
#testClassifier(DecisionTreeClassifier(), X, y)
final = testClassifier(BoostClassifier(DecisionTreeClassifier(), T=10), X, y,test=0.3,n=100)
print(final)


lawl = pd.DataFrame(final,columns={'classified':int})
def hate(x):
    return lastshit[x]


lawl['classified'] = lawl['classified'].apply(hate)
np.savetxt('45289.txt',lawl['classified'].values,fmt="%s")