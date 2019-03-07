import os
import math
import datetime
import pickle
import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import euclidean_distances, pairwise_distances
from collections import defaultdict, OrderedDict
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse
from cgi import parse_header, parse_multipart

class Recommender:
    # run initialization of all recommending models utilized in online evaluation
    # always keep only the resulting model, dictionary and rev_dict
    def save_obj(self, obj, name):
        with open('obj/' + name + '.pkl', 'wb') as f:
            pickle.dump(obj, f)

    def load_obj(self, name):
        with open('obj/' + name + '.pkl', 'rb') as f:
            return pickle.load(f)

    def mmr_objects_similarity(self, i, j, rev_dict):
        try:
            idi = self.cbDict[rev_dict[i]]
            idj = self.cbDict[rev_dict[j]]
            return self.dfCBSim[idi, idj]
        except:
            return 0

    def mmr_sorted(self, docs, lambda_, results, rev_dict, length):
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
        # print("enter to MMR")
        selected = OrderedDict()
        docs = set(docs)
        while (len(selected) < len(docs)) and (len(selected) < length):
            remaining = docs - set(selected)
            mmr_score = lambda x: lambda_ * results[x] - (1 - lambda_) * max(
                [self.mmr_objects_similarity(x, y, rev_dict) for y in set(selected) - {x}] or [
                    0])  # TODO: self.mmr_objects_similarity
            next_selected = self.argmax(remaining, mmr_score)
            selected[next_selected] = len(selected)
            # print(len(selected))
        return selected

    def argmax(self, keys, f):
        return max(keys, key=f)

    def init_recommending_models(self, algDesc):
        for alg in algDesc:
            if len(self.alg_models[(alg["name"], alg["params"])]) == 0:
                # load model to dictionary
                model = self.load_obj(alg["name"] + "_" + alg["params"] + "_model")
                if alg["name"] == "word2vec":
                    dictionary = self.load_obj(alg["name"] + "_" + alg["params"] + "_dict")
                    rev_dict = self.load_obj(alg["name"] + "_" + alg["params"] + "_revdict")
                else:
                    dictionary = self.load_obj(alg["name"] + "_dict")
                    rev_dict = self.load_obj(alg["name"] + "_revdict")
                self.alg_models[(alg["name"], alg["params"])] = [model, dictionary, rev_dict]
                # TODO: create models from full input data

    def __init__(self):
        self.alg_models = defaultdict(list)
        self.algorithm_descriptions = [
            {"name":"doc2vec", "params": "128_1", "aggregation": "last", "novelty": False, "diversity": True},
            {"name":"doc2vec", "params": "128_1", "aggregation": "temporal", "novelty": True, "diversity": False},
            {"name":"doc2vec", "params": "32_5", "aggregation": "mean", "novelty": False, "diversity": False},
            {"name":"doc2vec", "params": "32_5", "aggregation": "mean", "novelty": True, "diversity": False},
            {"name":"doc2vec", "params": "128_5", "aggregation": "max", "novelty": False, "diversity": True},
            {"name":"vsm", "params": "noSameObjects", "aggregation": "temporal", "novelty": False, "diversity": True},
            {"name":"vsm", "params": "sameAllowed", "aggregation": "mean", "novelty": False, "diversity": True},
            {"name":"vsm", "params": "sameAllowed", "aggregation": "window10", "novelty": False, "diversity": False},
            {"name":"word2vec", "params": "64_5", "aggregation": "mean", "novelty": True, "diversity": False},
            {"name":"word2vec", "params": "32_5", "aggregation": "temporal", "novelty": False, "diversity": True},
            {"name":"word2vec", "params": "128_3", "aggregation": "last", "novelty": False, "diversity": False},
            {"name":"word2vec", "params": "32_3", "aggregation": "window10", "novelty": False, "diversity": False}
        ]
        print("Volume of recommenders: {}".format(len(self.algorithm_descriptions)))

        self.init_recommending_models(self.algorithm_descriptions)

        dfValidDates = pd.read_csv("data/serialValidDates.csv", sep=";", header=0, index_col=0)
        dfValidDates.novelty_date = pd.to_datetime(dfValidDates.novelty_date)
        now = datetime.datetime.now()
        novelty_score = 1 / np.log((now - dfValidDates.novelty_date).dt.days + 2.72)
        # print(novelty_score)
        dfValidDates["noveltyScore"] = novelty_score
        self.dfValidDates = dfValidDates
        dct = defaultdict(int)
        self.noveltyDict = dfValidDates.noveltyScore.to_dict(into=dct)

        dfCBFeatures = pd.read_csv("data/serialCBFeatures.txt", sep=",", header=0, index_col=0)
        self.dfCBSim = 1 - pairwise_distances(dfCBFeatures, metric="cosine")

        cbNames = dfCBFeatures.index.values
        cbVals = range(len(cbNames))
        self.cbDict = dict(zip(cbNames, cbVals))

    def recommend(self, model, dictionary, rev_dict, userTrainData, userTrainLogDates, alg, rec, diversity, novelty,
                  allowedOIDs):

        # remove objects that are no longer valid TODO transform to keep only allowed OIDs
        if len(allowedOIDs) > 0:
            resultsValidity = [i for i in range(len(rev_dict)) if (rev_dict[i] in self.dfValidDates.index) and (
                self.dfValidDates.available_date[rev_dict[i]] > "2018-06-01") and (rev_dict[i] in allowedOIDs)]
        else:
            resultsValidity = [i for i in range(len(rev_dict)) if (rev_dict[i] in self.dfValidDates.index) and (
                self.dfValidDates.available_date[rev_dict[i]] > "2018-06-01")]

        #print(len(resultsValidity))
        try:
            # remove no longer known IDs
            trainModelIDs = list(map(dictionary.get, userTrainData))
            if (rec == "temporal") | (rec == "temporal3") | (rec == "temporal5") | (rec == "temporal10"):
                tw = [userTrainLogDates[i] for i in range(len(userTrainLogDates)) if trainModelIDs[i] is not None]

            trainModelIDs = list(filter(None.__ne__, trainModelIDs))
            userTrainData = list(map(rev_dict.get, trainModelIDs))

        except:
            print("Error")
            userTrainData = []
        #print(len(userTrainData))
        if (len(userTrainData) > 0):
            if (rec == "mean") | (rec == "max"):
                weights = [1.0] * len(userTrainData)
            elif rec == "last":
                # userTrainData = userTrainData[-1:]
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
                # userTrainData = userTrainData[-3:]
                trainModelIDs = trainModelIDs[-3:]
                weights = [float(i) for i in tw[-3:]]

            elif rec == "temporal5":
                # userTrainData = userTrainData[-5:]
                trainModelIDs = trainModelIDs[-5:]
                weights = [float(i) for i in tw[-5:]]

            elif rec == "temporal10":
                # userTrainData = userTrainData[-10:]
                trainModelIDs = trainModelIDs[-10:]
                weights = [float(i) for i in tw[-10:]]

            elif rec == "temporal":
                weights = [float(i) for i in tw]

            # print(trainModelIDs)
            # print(type(trainModelIDs[0]))
            embeds = model[trainModelIDs]
            if alg == "vsm":  # attributeCosineSim
                results = embeds
            else:
                results = 1 - pairwise_distances(embeds, model, metric="cosine")

            weights = np.asarray(weights).reshape((-1, 1))
            if rec == "max":
                results = np.max(results, axis=0)
            else:
                results = results * weights
                results = np.mean(results, axis=0)



            noveltyList = np.asarray(list(map(self.noveltyDict.get, [rev_dict[i] for i in range(len(rev_dict))])))
            noveltyList = noveltyList[resultsValidity]

            rdKeys = range(len(results))
            rdVals = [rev_dict[i] for i in resultsValidity]
            rev_dict_updated = dict(zip(rdKeys, rdVals))

            results = results[resultsValidity]

            if novelty == True:
                results = (0.8 * results) + (0.2 * noveltyList)
            if diversity == True:
                resultList = self.mmr_sorted(range(len(results)), 0.8, results, rev_dict_updated, 10)
            else:
                resultList = (-results).argsort()[0:20]
            return [rev_dict_updated[i] for i in resultList]

print('starting recommender...')
recommender = Recommender()

# HTTPRequestHandler class
class Reveal_HTTPServer_RequestHandler(BaseHTTPRequestHandler):

  def parse_POST(self):
        ctype, pdict = parse_header(self.headers['content-type'])
        if ctype == 'multipart/form-data':
            postvars = parse_multipart(self.rfile, pdict)
        elif ctype == 'application/x-www-form-urlencoded':
            length = int(self.headers['content-length'])
            postvars = parse_qs(
                self.rfile.read(length),
                keep_blank_values=1)
        else:
            postvars = {}
        return postvars

  def do_GET(self):
      self.send_response(200)
      # Send headers
      self.send_header('Content-type', 'text/html')
      self.end_headers()
      params = parse_qs(urlparse(self.path).query)
      print(params.get("uid", [""]))
      if params.get("uid", "") != "":  # it is a valid request, no favicon etc
          print("GET request")
          # postvars =
          # Send response status code
          #try:
          if True:
              uid = int(params["uid"][0])
              allowed_oids = [int(i) for i in params.get("allowed_oids", [""])[0].split(",") if len(i) > 0]
              visited_oids = [int(i) for i in params.get("visited_oids", [""])[0].split(",") if
                              len(i) > 0]  # from oldest to newest
              visits_datetime = params.get("visits_datetime", [""])[0].split(",")  # from oldest to newest
              print(visited_oids)
              #print(visits_datetime)
              #print(len(allowed_oids))

              now = datetime.datetime.now()
              # now = datetime.datetime(2018, 7, 20, 00, 00) #maybe put actual now
              visits_logDays = [
                  1 / (math.log(max([(now - datetime.datetime.strptime(i, '%Y-%m-%d %H:%M:%S')).days, 1])) + 0.1)
                  for i in visits_datetime]
              k = len(recommender.algorithm_descriptions)
              algorithmVariant = uid % k
              alg = recommender.algorithm_descriptions[algorithmVariant]
              model = recommender.alg_models[(alg["name"], alg["params"])][0]
              dictionary = recommender.alg_models[(alg["name"], alg["params"])][1]
              rev_dict = recommender.alg_models[(alg["name"], alg["params"])][2]
              #print(len(rev_dict))
              results = recommender.recommend(model, dictionary, rev_dict, visited_oids, visits_logDays, alg["name"],
                                              alg["aggregation"], alg["diversity"], alg["novelty"], allowed_oids)
              print(results)
              resultsTxt = ",".join([str(i) for i in results])

          #except:
          #    resultsTxt = "error"
          #    print ("error")
          message = resultsTxt
          self.send_response(200)
          # Send headers
          self.send_header('Content-type', 'text/html')
          self.end_headers()

          # store the query and response to the logfile
          # Send message back to client
          # response: coma separated top-20 recommended objects
          # on error return "error"
          with open("log.txt", "a") as f:
              f.write("{};{};{};{};{}\n".format(now, uid, visited_oids, (alg["name"], alg["params"], alg["aggregation"], alg["diversity"], alg["novelty"]), resultsTxt))
          # print(message)
          print(datetime.datetime.now() - now)
          # Write content as utf-8 data
          self.wfile.write(bytes(message, "utf8"))
      return

  def do_POST(self):
     #params = self.parse_POST()
     params = parse_qs(urlparse(self.path).query)
     if params.get("uid", "") != "": #it is a valid request, no favicon etc
        print("POST request")
        #postvars =
        # Send response status code
        try:
        #if True:
            uid = int(params["uid"][0])
            allowed_oids = [int(i) for i in params.get("allowed_oids",[""])[0].split(",") if len(i) > 0]
            visited_oids = [int(i) for i in params.get("visited_oids",[""])[0].split(",") if len(i) > 0] #from oldest to newest
            visits_datetime = params.get("visits_datetime",[""])[0].split(",")  # from oldest to newest

            now = datetime.datetime.now()
            #now = datetime.datetime(2018, 7, 20, 00, 00) #maybe put actual now
            visits_logDays = [1 / (math.log(max([(now - datetime.datetime.strptime(i, '%Y-%m-%d %H:%M:%S')).days, 1])) + 0.1)
                      for i in visits_datetime]
            k = len(recommender.algorithm_descriptions)
            algorithmVariant = uid % k
            alg = recommender.algorithm_descriptions[algorithmVariant]
            model = recommender.alg_models[(alg["name"], alg["params"])][0]
            dictionary = recommender.alg_models[(alg["name"], alg["params"])][1]
            rev_dict = recommender.alg_models[(alg["name"], alg["params"])][2]
            results = recommender.recommend(model, dictionary, rev_dict, visited_oids, visits_logDays, alg["name"], alg["aggregation"], alg["diversity"], alg["novelty"], allowed_oids)
            resultsTxt = ",".join([str(i) for i in results])

        except:
            resultsTxt = "error"
        message = resultsTxt
        self.send_response(200)
        # Send headers
        self.send_header('Content-type','text/html')
        self.end_headers()

        # store the query and response to the logfile
        # Send message back to client
        # response: coma separated top-20 recommended objects
        # on error return "error"
        with open("log.txt", "a") as f:
            f.write("{};{};{};{};{}\n".format(now, uid,visited_oids, (alg["name"], alg["params"], alg["aggregation"], alg["diversity"], alg["novelty"]), resultsTxt))
        #print(message)
        print(datetime.datetime.now() - now)
        # Write content as utf-8 data
        self.wfile.write(bytes(message, "utf8"))
     return

def run():
    print('starting server...')  # Server settings
    # Choose port 8080, for port 80, which is normally used for a http server, you need root access
    server_address = ('', 50000)
    httpd = HTTPServer(server_address, Reveal_HTTPServer_RequestHandler)
    print('running server...')
    httpd.serve_forever()

run()