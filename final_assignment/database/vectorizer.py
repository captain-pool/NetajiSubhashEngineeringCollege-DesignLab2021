import numpy as np
import spacy
from absl import logging


class Vectorize(object):

  NOUNS = {"PROPN", "NOUN"}

  def __init__(self, model="en_core_web_lg", **kwargs):
    logging.info(f"Loading {model} for vectorizing")
    if model.lower().strip() == "en_core_web_sm":
      logging.warn(
        "It recommended to use the default model "
        "for vectorizing strings, loading anyway"
      )
    self._model_name = model
    self._vmodel = spacy.load(self._model_name, **kwargs)
    self._vdim = self._vmodel("dummy").vector.shape[0]

  @property
  def dimension(self):
    return self._vdim

  @property
  def model(self):
    return self._vmodel

  @property
  def vocab(self):
    return self._vmodel.vocab

  def __call__(self, sentence):
    obj = self._vmodel(sentence)
    obj.vector = obj.vector[np.newaxis, :].astype(np.float32)
    return obj
