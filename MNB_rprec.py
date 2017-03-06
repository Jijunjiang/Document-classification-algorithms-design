import math, sys
import random, copy
import scipy.optimize

import loadUnigrams
        
def buildprobs(posKnown,negKnown,knownIDs):

    knownWords = set()
    totalWords = totalKnownWords = 0
    totalPosWords = totalNegWords = 0
    totalPosKnownWords = totalNegKnownWords = 0

    for fileID in fileIDs:
        count = sum(wordcounts[fileID].values())
        
        totalWords += count
        if fileID in posarticles:
            totalPosWords += count

            if fileID in posKnown:
                totalPosKnownWords += count
                totalKnownWords += count
                knownWords = knownWords.union(set(wordcounts[fileID].keys()))
        else:
            totalNegWords += count

            if fileID in negKnown:
                totalNegKnownWords += count
                totalKnownWords += count
                knownWords = knownWords.union(set(wordcounts[fileID].keys()))

    numKnownWords = len(knownWords)

    PC_est = float(totalPosKnownWords)/totalKnownWords
    PnC_est= float(totalNegKnownWords)/totalKnownWords #negtive prior probobility

    TotPwCMLE = TotPwnCMLE = 0

    numswapped = 0

    for word in vocab:
        wordArticles = set(MISSLE[word].keys())
        posWordArt = wordArticles.intersection(posarticles)
        negWordArt = wordArticles.difference(posWordArt)
        posWordKnownArt = posWordArt.intersection(knownIDs)
        negWordKnownArt = negWordArt.intersection(knownIDs)

        pWord = float(sum(MISSLE[word].values()))/totalWords

        posWordKnownOcc = float(sum(MISSLE[word][ID] for ID in posWordKnownArt))
        negWordKnownOcc = float(sum(MISSLE[word][ID] for ID in negWordKnownArt))

        # MNB -- Standard smoothing (... + 1) / (... + |V|)
        PwC_MNB[word] = float(1+posWordKnownOcc)/(len(vocab) + totalPosKnownWords)
        PwnC_MNB[word] = float(1+negWordKnownOcc)/(len(vocab) + totalNegKnownWords)

        # True values -- calculated over all articles, assumed to be perfect
        PwC_truth[word] = float(sum(MISSLE[word][ID] for ID in posWordArt))/totalPosWords
        PwnC_truth[word] = float(sum(MISSLE[word][ID] for ID in negWordArt))/totalNegWords
    
        if posWordKnownOcc == 0 and negWordKnownOcc == 0:
            PwC_MLE[word] = pWord
            PwnC_MLE[word] = pWord
        else:
            A = max(0.5,posWordKnownOcc)
            C = max(0.5,negWordKnownOcc)
           
            B = totalPosKnownWords - A   
            D = totalNegKnownWords - C

            L = PC_est/PnC_est #positive prior probobility / negtive prior probobility

            K = pWord/PnC_est
            
            epsilon = .01

            def f(x):
                return A/x+B/(x-1)+C*L/(L*x-K)+D*L/(L*x-K+1)
            try:
                x = scipy.optimize.newton(f,K/(2*L))
            except RuntimeError:
                x = -1

            try:
                if x < 0 or x > 1 or K-L*x < 0 or K-L*x > 1:
                    x = scipy.optimize.newton(f,K/(3*L))
                    
                if x < 0 or x > 1 or K-L*x < 0 or K-L*x > 1:
                    x = scipy.optimize.newton(f,K/(1.5*L))
                if x < 0 or x > 1 or K-L*x < 0 or K-L*x > 1:
                    x = scipy.optimize.newton(f,K/(4*L))            
            except RuntimeError:
                x = -1

            if x < 0 or x > 1 or K-L*x < 0 or K-L*x > 1:
                PwC_MLE[word] = PwC_MNB[word]
                PwnC_MLE[word] = PwnC_MNB[word]
                numswapped += 1
            else:
                PwC_MLE[word] = x
                PwnC_MLE[word] = K-L*x
                
        TotPwCMLE += PwC_MLE[word]
        TotPwnCMLE += PwnC_MLE[word]

    ApprxNormErr = 0
    for word in vocab:        
        PwC_MLEnorm[word] = PwC_MLE[word] / TotPwCMLE
        PwnC_MLEnorm[word] = PwnC_MLE[word] / TotPwnCMLE

# Evaluate any conditional probability tables on testIDs
# Requires: P(w|C) and P(w|~C) tables, an estimate of P(C), and 
#   list of testIDs to evaluate model on
# Returns: TP/FP/FN/TN counts, R-Precision of the positive class,
#   and macro-averaged positive and negative R-Precision
def evaluate(pWordClassTable, pWordNotClassTable, pDocClass, testIDs):
    scores = []
    truepos = trueneg = falsepos = falseneg = 0
    
    # For each test document, evaluate MNB model using P(w|C) and P(w|~C) tables
    for fileID in testIDs:
        pClass = math.log(pDocClass)
        pNotClass = math.log(1 - pDocClass)

        for word in wordcounts[fileID]:
            pWordClass = pWordClassTable[word]
            pWordNotClass = pWordNotClassTable[word]

            try:
                pClass += wordcounts[fileID][word]*math.log(pWordClass)
                pNotClass += wordcounts[fileID][word]*math.log(pWordNotClass)
            except ValueError:
                print ""
                print ""
                print "Error while taking log of probabilities:"
                print "Word:", word
                print "P(w| C):", pWordClass
                print "P(w|~C):", pWordNotClass

                input("(Paused. Hit Enter to crash)")

		# Compare P(d \in C) and P(d \in ~C) and determine if guess was correct
        if pClass > pNotClass:
            if className in labels[fileID]:
                truepos += 1
            else:
                falsepos += 1
        else:
            if className in labels[fileID]:
                falseneg += 1
            else:
                trueneg += 1

        scores.append([pClass-pNotClass,fileID, className in labels[fileID]])
    scores.sort(reverse=True)

    cutoff = len(set(testIDs).intersection(posarticles))
    tp = fp = fn = tn = 0
    for pair in scores[:cutoff]:
        if className in labels[pair[1]]:
            tp += 1
        else:
            fp += 1
    for pair in scores[cutoff:]:
        if className in labels[pair[1]]:
            fn += 1
        else:
            tn += 1
    try:
        rPrec_Pos = tp / float(tp + fp)
    except ZeroDivisionError:
        rPrec_Pos = 0
    try:
        rPrec_Neg = tn / float(tn + fn)
    except ZeroDivisionError:
        rPrec_Neg = 0
    avgrPrec = (rPrec_Pos+rPrec_Neg)/2

    return (truepos,falsepos,falseneg,trueneg,rPrec_Pos,avgrPrec)

# To show how models scale, create training splits of the largest training set size.
# Training on smaller sets of data should only use the first N of the #maxtrain# instances. 
# Note: First document is always negative, second is always positive --
#       otherwise, it's impossible to learn a classification one way or another
# Returns: Ordered list of #numIDs# IDs from global set/list/tuple fileIDs
def generateTrainingOrder(numIDs):
    # Select one positive and one negative ID at random
    firstPos = random.choice(tuple(posarticles))
    firstNeg = random.choice(tuple(negarticles))

    # Copy fileIDs and shuffle order -- no guarantee of order, so naive approach is fine
    trainingOrder = list(fileIDs)
    random.shuffle(trainingOrder)
    
    # Move random pos and neg IDs to front
    trainingOrder.remove(firstPos)
    trainingOrder.remove(firstNeg)
    trainingOrder.insert(0,firstPos)
    trainingOrder.insert(0,firstNeg)

    return trainingOrder[:numIDs]

# Load training splits from a file
#   Format: "fileID1, ..., fileIDN\n" per line (each line is one split)
#   Returns: List of splits; Each split is an ordered list of IDs
def loadsplits(filename, splitchar):
    splits = []
    f = open(filename)
    for line in f:
        partition = line.split(splitchar)
        split = []
        for i in xrange(len(partition)):
            try:
                split.append(int(float(partition[i])))
            except ValueError:
                pass
        splits.append(split)
    return splits

# Train models
def runtest(trainsizes, splitFile = None, trainIDs = None):
    # Determine which documents belong to testIDs and trainIDs based on settings
    if loadSplits:
        trainingOrder = copy.copy(trainIDs)
        testIDs = set(fileIDs).difference(set(trainingOrder))
    elif writeSplits:
        trainingOrder = generateTrainingOrder(max(trainsizes))
        testIDs = set(fileIDs).difference(set(trainingOrder))

        f = open(splitFile,'a')
        for testID in trainingOrder:
            f.write("%d " % testID)
        f.write("\n")
        f.close()
    else:
        trainingOrder = generateTrainingOrder(max(trainsizes))
        testIDs = set(fileIDs).difference(set(trainingOrder))
    
    for sampleSize in trainSizes:
        knownIDs = set(trainingOrder[:sampleSize])
        
        posKnown = posarticles.intersection(knownIDs)
        negKnown = negarticles.intersection(knownIDs)

        buildprobs(posKnown,negKnown,knownIDs)
        
        pDocClass = len(posKnown)/float(sampleSize)
        
        result["MNB" + str(sampleSize)] = map(lambda x,y:x+y, result["MNB" + str(sampleSize)], evaluate(PwC_MNB, PwnC_MNB, pDocClass, testIDs))
        result["MLE" + str(sampleSize)] = map(lambda x,y:x+y, result["MLE" + str(sampleSize)], evaluate(PwC_MLE, PwnC_MLE, pDocClass, testIDs))
        result["MLE_Norm" + str(sampleSize)] = map(lambda x,y:x+y, result["MLE_Norm" + str(sampleSize)], evaluate(PwC_MLEnorm, PwnC_MLEnorm, pDocClass, testIDs))

        rPrec_counts[sampleSize] += len(set(testIDs).intersection(posarticles))
        
# RUNTIME SETTINGS
numIters = 5
trainSizes = [10,100,1000]#,10,20,30,40,50,75,100,150,200,300,400,500,750,1000]

identifiers = ["MNB","MLE","MLE_Norm"] # Methods to train
loadSplits = False    # Load splitfile to specify train/test splits
writeSplits = True# Write splitfile when done

## Amazon Sentiment Data -- Lucas & Downey
#for topicName in ["toys_games","kitchen","dvd","electronics"]:
for topicName in ["toys_games"]:
    loadUnigrams.load("lucas_downey_data/" + topicName + ".processed.review")
    splitfile = "lucas_downey_data/splits/" + topicName + "_splits.txt"

## Amazon Sentiment Data -- Zhao et al.
#for topicName in ["toys_games","kitchen","dvd_lucas","electronics"]:
    loadUnigrams.load_arff("zhao_data/" + topicName + ".arff")
    
    className = "positive"
    print "class:", topicName
    print "instances:", len(wordcounts)
    print "words processed:", len(MISSLE)
    if loadSplits:
        splits = loadsplits(splitfile, " ")
        print "splits loaded"
    elif writeSplits:
        f = open(splitfile,'w')
        f.close()
    
    # Converting to list gives each word a unique ID; better for iteration
    vocab = list(vocabulary)
    print "total tokens:",sum([sum(MISSLE[word].values()) for word in vocab]) 

    # Record which files belong to the class; sets chosen for quick intersection operations
    posarticles = set()
    negarticles = set()
    for fileID in fileIDs:
        if className in labels[fileID]:
            posarticles.add(fileID)
        else:
            negarticles.add(fileID)
            
    print "%d positive, %d negative articles (%.2f%% positive)" % (len(posarticles), len(negarticles), float(len(posarticles))*100/len(fileIDs))

    # Initialize lookup tables for Conditional Word Probabilities
    #   P(w | C) and P(w | ~C) table for each model
    PwC_MNB = {}      # Multinomial Naive Bayes
    PwnC_MNB = {}
    PwC_MLE = {}      # Unnormalized MNB-FM
    PwnC_MLE = {}
    PwC_MLEnorm = {}  # MNB-FM
    PwnC_MLEnorm = {}
    PwC_truth = {}    # Truth (uses all training and test documents)
    PwnC_truth = {}

    result = {}       # Records evaluation results for different models over each iteration
    rPrec_counts = {} # Number of unknown positives -- necessary for R-Precision calculations

	# Initialize record-keeping objects
    for trainSize in trainSizes:
        for identifier in identifiers:
            result[identifier + str(trainSize)] = (0,0,0,0,0,0)
        rPrec_counts.setdefault(trainSize,0)
    
    # During each iteration, train and evaluate each model
    for iteration in xrange(numIters):
##        print "(%d)" % iteration,
        if loadSplits:
            runtest(trainSizes, trainIDs=splits[iteration])
        elif writeSplits:
            runtest(trainSizes, splitFile=splitfile)
        else:
            runtest(trainSizes)

    # Output evaluation results
    for ident in identifiers:
        print ident
        for sampleSize in trainSizes:
            print sampleSize,
        print ""
        for i in xrange(6):
            for sampleSize in trainSizes:
                print result[ident+str(sampleSize)][i]/float(numIters),
            print ""

    # Output R-Precision evaluations
    print "rPrec_sizes",
    for sampleSize in trainSizes:
        print float(rPrec_counts[sampleSize]) / numIters,
    print ""
