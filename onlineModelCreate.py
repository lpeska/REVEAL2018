import numpy as np
import pandas as pd
import doc2vec
import word2vec
import rank_metrics
import datetime
import pickle
from collections import defaultdict, OrderedDict
from sklearn.metrics.pairwise import euclidean_distances, pairwise_distances
from sklearn.metrics import *
from sklearn.preprocessing import MinMaxScaler
from ast import literal_eval

listOfAlgs = [ "attributeCosineSim", "doc2vec", "word2vec"]
embedSizes = [32, 64, 128]
windowSizes = [1, 3, 5]
#SELECT userID, min(lastModified), max(lastModified) FROM `userEvents` group by userID having max(lastModified) >= "2018-06-01" and min(lastModified) < "2018-06-01"

dfValidDates = pd.read_csv("data/serialValidDates.csv", sep=";", header=0, index_col=0)
dfValidDates.novelty_date = pd.to_datetime(dfValidDates.novelty_date)
now = datetime.datetime.now()
novelty_score = 1 / np.log((now - dfValidDates.novelty_date).dt.days + 2.72)
#print(novelty_score)
dfValidDates["noveltyScore"] = novelty_score
dct = defaultdict(int)
noveltyDict = dfValidDates.noveltyScore.to_dict(into=dct)

dfCBFeatures = pd.read_csv("data/serialCBFeatures.txt", sep=",", header=0, index_col=0)
dfCBSim = 1 - pairwise_distances(dfCBFeatures, metric="cosine")

#denote the
dfCBSimNoSame = np.copy(dfCBSim)
np.fill_diagonal(dfCBSimNoSame, 0.0)

cbNames = dfCBFeatures.index.values
cbVals = range(len(cbNames))
rev_cbDict = dict(zip(cbVals, cbNames))
cbDict = dict(zip(cbNames, cbVals))

print(dfCBSim.shape)
print(dfCBSim[0:5,0:5])

df = pd.read_csv("data/serialTexts.txt", sep=";", header=0, index_col=0)
d2v_names = df.index.values
d2v_vals = range(len(d2v_names))

rev_dict_d2v = dict(zip(d2v_vals, d2v_names))
dict_d2v = dict(zip(d2v_names, d2v_vals))



#print(dict_d2v)
#print(rev_dict_d2v)
print(len(d2v_names), len(d2v_vals), len(dict_d2v), len(rev_dict_d2v))

testSet = pd.read_csv("data/test_data_wIndex.txt", sep=",", header=0, index_col=0)
testSet["oids"] = testSet.strOID.str.split()
trainSet = pd.read_csv("data/train_data_wIndex.txt", sep=",", header=0, index_col=0)
trainSet["oids"] = trainSet.strOID.str.split()

trainTimeWeight = pd.read_csv("data/serialLogDays.txt", sep=",", header=0, index_col=0, converters={"logDays": literal_eval})
print(trainTimeWeight.head())
#trainTimeWeight["weights"] = trainTimeWeight.logDays.str.split()

def mmr_objects_similarity(i, j, rev_dict):
    try:
        idi = cbDict[rev_dict[i]]
        idj = cbDict[rev_dict[j]]
        return dfCBSim[idi, idj]
    except:
        return 0

def mmr_sorted(docs, lambda_, results, rev_dict, length):
    """Sort a list of docs by Maximal marginal relevance

	Performs maximal marginal relevance sorting on a set of
	documents as described by Carbonell and Goldstein (1998)
	in their paper "The Use of MMR, Diversity-Based Reranking
	for Reordering Documents and Producing Summaries"

    :param docs: a set of documents to be ranked
				  by maximal marginal relevance
    :param q: query to which the documents are results
    :param lambda_: lambda parameter, a float between 0 and 1
    :return: a (document, mmr score) ordered dictionary of the docs
			given in the first argument, ordered my MMR
    """
    #print("enter to MMR")
    selected = OrderedDict()
    docs = set(docs)
    while (len(selected) < len(docs)) and (len(selected) < length):
        remaining = docs - set(selected)
        mmr_score = lambda x: lambda_ * results[x] - (1 - lambda_) * max(
            [mmr_objects_similarity(x, y, rev_dict) for y in set(selected) - {x}] or [0])
        next_selected = argmax(remaining, mmr_score)
        selected[next_selected] = len(selected)
        #print(len(selected))
    return selected


def argmax(keys, f):
    return max(keys, key=f)

def user_novelty_at_n(rankedIDs, trainModelIDs, n):
    return np.sum([1 for i in rankedIDs[0:n] if i in trainModelIDs])/n

def prec_at_n(rankedRelevance, n):
    return np.sum(rankedRelevance[0:n])/n

def meanNovelty_at_n(noveltyList, n):
    return np.sum(noveltyList[0:n])/n

def rec_at_n(rankedRelevance, n):
    return np.sum(rankedRelevance[0:n])/np.sum(rankedRelevance)

def ild_at_n(idx, rev_dict,  n):
    divList = []
    for i in idx[0:n]:
        for j in idx[0:n]:
            try:
                idi = cbDict[rev_dict[i]]
                idj = cbDict[rev_dict[j]]
                if i != j:
                    divList.append(1-dfCBSim[idi, idj])
            except:
                pass
    return np.mean(divList)

def evalResults(results, trueRelevance, noveltyList, trainModelIDs, rev_dict, uid, alg, params, rec, outFile, diversity, novelty):
    params = [str(i) for i in params]
    #calculate rating precision
    mmScaler = MinMaxScaler(copy=True)
    results = mmScaler.fit_transform(results.reshape(-1,1))
    results = results.reshape((-1,))
    r2Sc = r2_score(trueRelevance, results)
    mae = mean_absolute_error(trueRelevance, results)


    #calculate ranking scores
    idx = (-results).argsort()

    if diversity == "yes":
        reranked = mmr_sorted(range(len(results)), 0.8, results, rev_dict, 10)
        idx1 = [k for k, v in reranked.items()]
        idx2 = [i for i in idx if i not in idx1]
        idx1.extend(idx2)
        idx = idx1

    rankedRelevance = trueRelevance[idx]
    rankedNovelty = noveltyList[idx]

    #print(rankedRelevance)

    map = rank_metrics.average_precision(rankedRelevance)
    aucSc = roc_auc_score(trueRelevance, results)
    nDCG10 = rank_metrics.ndcg_at_k(rankedRelevance,10)
    nDCG100 = rank_metrics.ndcg_at_k(rankedRelevance, 100)
    nDCG = rank_metrics.ndcg_at_k(rankedRelevance, len(rankedRelevance))

    p5 = prec_at_n(rankedRelevance, 5)
    r5 = rec_at_n(rankedRelevance, 5)
    n5 = meanNovelty_at_n(rankedNovelty, 5)
    un5 = user_novelty_at_n(idx, trainModelIDs, 5)
    ild5 = ild_at_n(idx, rev_dict, 5)
    p10 = prec_at_n(rankedRelevance, 10)
    r10 = rec_at_n(rankedRelevance, 10)
    n10 = meanNovelty_at_n(rankedNovelty, 10)
    ild10 = ild_at_n(idx, rev_dict, 10)
    un10 = user_novelty_at_n(idx, trainModelIDs, 10)

    mrr = rank_metrics.mean_reciprocal_rank([rankedRelevance])


    #print((uid, alg, ",".join(params), rec, r2Sc, mae, map, aucSc, mrr, p5, p10, r5, r10, nDCG10, nDCG100, nDCG))

    txt = "%s;%s;%s;%s;%s;%s;%.6f;%.6f;%.6f;%.6f;%.6f;%.6f;%.6f;%.6f;%.6f;%.6f;%.6f;%.6f;%.6f;%.6f;%.6f;%.6f;%.6f;%.6f\n"%(uid, alg, ",".join(params), rec, diversity, novelty, r2Sc, mae, map, aucSc, mrr, p5, p10, r5, r10, nDCG10, nDCG100, nDCG, n5, n10, un5, un10, ild5, ild10)
    outFile.write(txt)
    return(r2Sc, mae, map, aucSc, mrr, p5, p10, r5, r10, nDCG10, nDCG100, nDCG, n5, n10, ild5, ild10)



def eval(model, dictionary, rev_dict, testSet, trainSet, alg, params, resultsFile):
    recsysStrategies = ["temporal",  "temporal3", "temporal5", "temporal10", "mean", "max", "last", "window3", "window5", "window10"] #, "diversity", "novelty"

    # remove objects that are no longer valid
    resultsValidity = [i for i in range(len(rev_dict)) if (rev_dict[i] in dfValidDates.index) and (dfValidDates.available_date[rev_dict[i]] > "2018-06-01")]
    #print(resultsValidity)

    for rec in recsysStrategies:
        for uid in testSet.index:
            # print(uid)
            # print(dictionary)
            # print(rev_dict)
            # exit()
            try:
                userTrainData = [int(i) for i in trainSet.oids[uid]]
                userTestData = [int(i) for i in testSet.oids[uid]]

                # remove no longer known IDs
                trainModelIDs = list(map(dictionary.get, userTrainData))
                if (rec == "temporal") |(rec == "temporal3") |(rec == "temporal5") |(rec == "temporal10"):
                    tw = trainTimeWeight.logDays[uid]
                    tw = [tw[i] for i in range(len(tw)) if trainModelIDs[i] is not None]

                trainModelIDs = list(filter(None.__ne__, trainModelIDs))
                userTrainData = list(map(rev_dict.get, trainModelIDs))

                testModelIDs = list(map(dictionary.get, userTestData))
                testModelIDs = list(filter(None.__ne__, testModelIDs))
                userTestData = list(map(rev_dict.get, testModelIDs))


            except:
                print("Error for user " + str(uid))
                userTrainData = []
                userTestData = []
            # print(len(userTrainData), len(userTestData))
            if (len(userTrainData) > 0) & (len(userTestData) > 0):

                trueRelevance = np.zeros(len(dictionary.keys()), dtype=int)
                trueRelevance[testModelIDs] = 1


                if (rec == "mean") | (rec == "max"):
                    weights = [1.0] * len(userTrainData)
                elif rec == "last":
                    userTrainData = userTrainData[-1:]
                    trainModelIDs = trainModelIDs[-1:]
                    weights = [1.0]
                elif rec == "window3":
                    userTrainData = userTrainData[-3:]
                    trainModelIDs = trainModelIDs[-3:]
                    weights = [1 / len(userTrainData) * i for i in range(1, (len(userTrainData) + 1))]
                elif rec == "window5":
                    userTrainData = userTrainData[-5:]
                    trainModelIDs = trainModelIDs[-5:]
                    weights = [1 / len(userTrainData) * i for i in range(1, (len(userTrainData) + 1))]
                elif rec == "window10":
                    userTrainData = userTrainData[-10:]
                    trainModelIDs = trainModelIDs[-10:]
                    weights = [1 / len(userTrainData) * i for i in range(1, (len(userTrainData) + 1))]

                elif rec == "temporal3":
                    #userTrainData = userTrainData[-3:]
                    trainModelIDs = trainModelIDs[-3:]
                    weights = [float(i) for i in tw[-3:]]
                    #weights = [1 / len(userTrainData) * i for i in range(1, (len(userTrainData) + 1))]
                elif rec == "temporal5":
                    #userTrainData = userTrainData[-5:]
                    trainModelIDs = trainModelIDs[-5:]
                    weights = [float(i) for i in tw[-5:]]
                    #weights = [1 / len(userTrainData) * i for i in range(1, (len(userTrainData) + 1))]
                elif rec == "temporal10":
                    #userTrainData = userTrainData[-10:]
                    trainModelIDs = trainModelIDs[-10:]
                    weights = [float(i) for i in tw[-10:]]
                    #weights = [1 / len(userTrainData) * i for i in range(1, (len(userTrainData) + 1))]
                elif rec == "temporal":
                    weights = [float(i) for i in tw]
                    #weights = [1 / len(userTrainData) * i for i in range(1, (len(userTrainData) + 1))]


                # print(trainModelIDs)
                # print(type(trainModelIDs[0]))
                embeds = model[trainModelIDs]
                # print(embeds.shape)
                if alg == "attributeCosineSim":
                    results = embeds
                else:
                    results = 1 - pairwise_distances(embeds, model, metric="cosine")

                weights = np.asarray(weights).reshape((-1, 1))
                if rec == "max":
                    results = np.max(results, axis=0)
                else:
                    results = results * weights
                    results = np.mean(results, axis=0)
                # print(results.shape, np.sum(trueRelevance))
                results = results[resultsValidity]
                trueRelevance = trueRelevance[resultsValidity]

                noveltyList = np.asarray(list(map(noveltyDict.get, [rev_dict[i] for i in range(len(rev_dict))])))
                noveltyList = noveltyList[resultsValidity]
                trainModelIDs = [i for i in trainModelIDs if i in resultsValidity]

                rdKeys = range(len(results))
                rdVals = [rev_dict[i] for i in resultsValidity]
                rev_dict_updated = dict(zip(rdKeys, rdVals))

                if (np.sum(trueRelevance) > 0):
                    resultMetrics = evalResults(results, trueRelevance, noveltyList, trainModelIDs, rev_dict_updated, uid, alg, params, rec, resultsFile, "no", "no")
                    resultMetrics = evalResults(results, trueRelevance, noveltyList, trainModelIDs, rev_dict_updated, uid, alg, params, rec, resultsFile, "yes", "no")
                    # enhance novelty as a (1 + nov_score) re-ranking to the results list
                    results = (0.8* results)  + (0.2*noveltyList)
                    resultMetrics = evalResults(results, trueRelevance, noveltyList, trainModelIDs, rev_dict_updated, uid, alg, params, rec, resultsFile,"no","yes")
                    resultMetrics = evalResults(results, trueRelevance, noveltyList, trainModelIDs, rev_dict_updated, uid, alg, params, rec, resultsFile, "yes","yes")


def save_obj(obj, name):
    with open('objAll/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f)


def load_obj(name):
    with open('objAll/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


with open("resultsWithNovDiv_32_0dot01Temporal.csv","w") as resultsFile:
    resultsFile.write("uid;alg;params;recAlg;noveltyEnhance;diversityEnhance;r2Score;mae;map;aucScore;mrr;p5;p10;r5;r10;nDCG10;nDCG100;nDCGFull;novelty5;novelty10;user_novelty5;user_novelty10;ild5;ild10\n")
    for alg in listOfAlgs:
        if alg == "word2vec":
            for e in embedSizes:
                for w in windowSizes:
                    model, rev_dict, dictionary = word2vec.word2vecRun(w,e)
                    dictionary = dict([((int(i),j) if i !="RARE" else (-1,j)) for i,j in dictionary.items() ])
                    rev_dict = dict(zip(dictionary.values(), dictionary.keys()))
                    #store models

                    save_obj(model, "word2vec_{0}_{1}_model".format(e,w))
                    save_obj(dictionary, "word2vec_{0}_{1}_dict".format(e, w))
                    save_obj(rev_dict, "word2vec_{0}_{1}_revdict".format(e, w))

                    #print("eval W2V")
                    #eval(model, dictionary, rev_dict, testSet, trainSet, alg, (e, w), resultsFile)
        elif alg == "doc2vec":
            for e in embedSizes:
                for w in windowSizes:
                    model = doc2vec.doc2vecRun(w,e)
                    rev_dict = rev_dict_d2v
                    dictionary = dict_d2v
                    # store models

                    save_obj(model, "doc2vec_{0}_{1}_model".format(e, w))
                    save_obj(dictionary, "doc2vec_dict")
                    save_obj(rev_dict, "doc2vec_revdict")

                    #print("eval D2V")
                    #eval(model, dictionary, rev_dict, testSet, trainSet, alg, (e,w), resultsFile)
        else:
            #TODO get CB data

            rev_dict = rev_cbDict
            dictionary = cbDict

            save_obj(dictionary, "vsm_dict")
            save_obj(rev_dict, "vsm_revdict")

            for same in ["sameAllowed", "noSameObjects"]:
                if same == "sameAllowed":
                    model = dfCBSim
                    save_obj(model, "vsm_{0}_model".format(same))
                else:
                    model = dfCBSimNoSame
                    save_obj(model, "vsm_{0}_model".format(same))
                #print("eval CB")
                #eval(model, dictionary, rev_dict, testSet, trainSet, alg, [same], resultsFile)
