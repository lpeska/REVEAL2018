# Off-line vs. On-line Evaluation of Recommender Systems in Small E-commerce - REVEAL 2018
This repository contains source codes, raw data and complete results of the paper "Off-line vs. On-line Evaluation of Recommender Systems in Small E-commerce", submitted to REVEAL (RecSys) 2018

The approach evaluates word2vec for collaborative information on the sequence of user's visits, doc2vec model for assessing similarity based on objects textual description and cosine similarity to determine similar objects based on their CB attributes. Furthermore, diversity and novelty enhancements and several variants of user's history processing evaluated.

- folder data/ contains source datasets originating from a medium-sized travel agency

- folder results/ contains off-line evaluation results

- folder onlineResults/ contains data acquired from evaluation on the production server

- run offLineEval.py to get the off-line results.

## Abstract
In this paper, we present our work in progress towards comparing on-line and off-line evaluation metrics in the context of small e-commerce recommender systems. Recommending on small e-commerce enterprises are rather challenging due to the lower volume of interactions and low user loyalty, rarely extending beyond a single session. On the other hand, we usually have to deal with lower volumes of objects, which are easier to discover by users through various browsing/searching GUIs.

The main goal of this paper is to determine applicability of off-line evaluation metrics in learning true usability of recommender systems (evaluated on-line in A/B testing). In total 800 variants of recommending algorithms were evaluated off-line w.r.t. 18 metrics covering rating-based, ranking-based, novelty and diversity evaluation. The off-line results were afterwards compared with on-line evaluation of 12 selected recommender variants.

Off-line results shown a great variance in performance w.r.t. different metrics with the Pareto front covering 68\% of the approaches. On-line metrics correlates positively with ranking-based metrics (MRR, nDCG), while too high values of diversity and novelty had a negative impact on the on-line results. We further train two regressors to predict on-line results based on the off-line metrics and estimate performance of recommenders not evaluated in A/B testing directly.
