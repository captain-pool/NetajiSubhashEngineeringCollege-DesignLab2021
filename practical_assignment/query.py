import pathlib
import _pickle as pickle
import json
import numpy as np
import re
import tqdm
import sys
import aspect_based_sentiment_analysis as absa
import collections
import database
import constants
import hydra


def read_data(ifile=pathlib.Path("data.json"), lfile=pathlib.Path("data.pickle")):
  D = collections.defaultdict(list)
  print("[read_data()]")
  pattern = re.compile("\d+")
  if lfile.exists():
    with open(lfile, "rb") as f:
      D = pickle.load(f)
  else:
    with open(ifile, "r") as f:
      for line in tqdm.tqdm(f):
        row = json.loads(line)
        rating = float(pattern.match(row["Rating"] or "0").group(0))
        review = row["ReviewText"]
        org = row["organisation"]
        D[org].append([rating, review])
    lfile.parent.mkdir(parents=True, exist_ok=True)
    print("[read_data()] no pickle found. dumping a new one.")
    with open(lfile, "wb") as w:
      pickle.dump(D, w)
  return D


def get_aspectdb(aspects, vct, dbpath=pathlib.Path("db/aspects"), update=False):
  print("[get_aspectdb()]")
  dbpath.parent.mkdir(parents=True, exist_ok=True)
  db = database.VectorDB(str(dbpath), vct.dimension).open()
  if not db.initialized or update:
    for aspect in tqdm.tqdm(aspects):
      db.insert(aspect, vct(aspect).vector)
    db.write()
  return db

def inference(model, row, keywords):
  scores = []
  rating, review = row
  if review:
    try:
      completed_tasks = model(review, aspects=keywords)
      scores = [float(task.sentiment) - 1 for task in completed_tasks]
    except:
      scores = []
  # A review without comment, but with stars
  # contribute equally to all aspects
  if not scores:
    scores = [rating / constants.TOTAL_STARS for _ in keywords]
  scores.append(rating)
  return scores


def reduce(scores):
  scores = np.asarray(scores).mean(axis=0).flatten()
  return {"keyword": scores[:-1] * 10, "stars": scores[-1]}


def infer(model, reviews, keywords):
  scores = []
  for review in tqdm.tqdm(reviews, position=1, leave=False):
    s = inference(model, review, keywords)
    scores.append(s)
  return reduce(scores)


def get_scoredb(
    model,
    D,
    keywords,
    update=False,
    dbpath=pathlib.Path("db/scores.pkl")):
  dbpath.parent.mkdir(parents=True, exist_ok=True)
  scoredb = collections.defaultdict(dict)
  if dbpath.exists():
    with open(dbpath, "rb") as f:
      scoredb = pickle.load(f)
    if not update:
      return scoredb
  if not isinstance(keywords, list):
    keywords = [keywords]
  for org in tqdm.tqdm(D, position=0):
    reviews = D[org]
    scores = infer(model, reviews, keywords)
    if not update:
      scoredb[org]["stars"] = scores["stars"]
      scores = {k: v for k, v in zip(keywords, scores["keyword"])}
      scoredb[org]["scores"] = scores
    else:
      scoredb[org]['scores'].update(
          {k: v for k, v in zip(keywords, scores['keyword'])})

  with open(dbpath, "wb") as f:
    pickle.dump(scoredb, f)
  return scoredb

def get_vocabdb(vct, dbpath=pathlib.Path("db/vocab")):
  print("[get_vocabdb()]")
  vocabdb = database.VectorDB(dbpath, vct.dimension).open()
  if not vocabdb.initialized:
    for word in tqdm.tqdm(vct.vocab.strings):
      if word.isalnum():
        vocabdb.insert(word, vct.vocab[word].vector)
    vocabdb.write()
  return vocabdb

def search_aspect(qaspect, vct, vocabdb, aspectdb):
  def search_matching_vocab_fn(aspectvector):
    return vocabdb.nearest(aspectvector)[0]
  if not qaspect.lower() in aspectdb.keys():
    aspectvct = vct(qaspect).vector
    aspect_closest = aspectdb.nearest(aspectvct)[0]
    vocab_closest_aspect = search_matching_vocab_fn(
        aspectdb.search_vector(
          aspectdb.nearest(aspectvct)[0]))
    qaspect = aspect_closest
    # Synonym search required. issue: Spacy Brown Cluster not working
    # find vocab entries for query aspect and closest aspect
    # if they do not belong to the same cluster, they are not the same
    if aspect_closest.lower() != vocab_closest_aspect.lower():
      return False, None
  return True, qaspect




@hydra.main(config_path="config", config_name="config")
def main(config):
  root = pathlib.Path(hydra.utils.get_original_cwd())
  D = read_data(root / config.datapath.input, root / config.datapath.load)
  model = absa.load(config.models.bert)
  vct = database.Vectorize(config.models.vectorizer)
  aspects = constants.ASPECTS
  scoredb = get_scoredb(model, D, aspects, dbpath=root / config.dbpath.score)
  aspectsdb = get_aspectdb(aspects, vct, dbpath=root / config.dbpath.aspects)
  vocabdb = get_vocabdb(vct, dbpath=root / config.dbpath.vocab)
  while True:
    org = input("enter org (OR 'q<RETURN>' to exit): ").strip().lower()
    if org == "q":
      break
    if org not in D:
      print('%s not an organization' % org)
      continue
    query_aspects = input("enter queries: ").strip().split(", ")
    for qaspect in query_aspects:
      present, aspect = search_aspect(qaspect, vct, vocabdb, aspectsdb)
      if not present:
        scoredb = get_scoredb(
            model,
            D,
            qaspect,
            dbpath=root / config.dbpath.score,
            update=True)
      print(
        "%s: %0.1f / 10 \t stars: %0.1f / 5"
        % (qaspect, scoredb[org]["scores"][aspect], scoredb[org]["stars"])
      )


if __name__ == "__main__":
  main()
