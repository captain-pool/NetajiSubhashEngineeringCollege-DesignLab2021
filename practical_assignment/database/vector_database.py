"""
 Similarity Database Module.
 Classes:
  - Database

 Author: @captain-pool
"""
import os

import faiss
import numpy as np
import _pickle as pickle
import orderedset


class VectorDB:
  """
  Database for storing Vectors and searching them by
  similarity.
  """

  def __init__(self, db_path, vector_dim):
    """
    Creates Database Instance.
    Args:
     db_path (str): Path to database storage folder
     vector_dim (int): Dimension of the vectors
    """
    name = os.path.basename(db_path)
    path = os.path.normpath(db_path)
    self._init = None
    self.payload__ = orderedset.OrderedSet()
    self.np_payload = None
    self.np_prev_hash = None
    if not os.path.exists(path) or not os.path.isdir(path):
      os.makedirs(path)
    self._index_file = os.path.join(path, "%s.index" % name)
    self._payload_path = os.path.join(path, "%s.payload" % name)
    self._inv_payload_path = os.path.join(path, "%s.invpayload" % name)
    self._vector_dim = vector_dim

  @property
  def initialized(self):
    if self._init is None:
      raise OSError("Database not opened")
    return self._init

  def __len__(self):
    if self._init:
      return len(self.payload__)
    return 0

  def __repr__(self):
    result = []
    result.append("Index \t| Path \t| Vector")
    result.append("-" * (len("Index t| Path | Vector") + 2 * 8))
    for idx, data in enumerate(self.payload__):
      result.append(f"{idx} \t| {data} \t| <float32 vector representation>")
    return "\n".join(result)

  def open(self):
    """
    Opens the database for reading and writing
    """
    self._init = True
    if os.path.exists(self._index_file) and os.path.exists(self._payload_path):
      self._index = faiss.read_index(self._index_file)
      with open(self._payload_path, "rb") as f:
        self.payload__ = orderedset.OrderedSet(
          np.load(f, allow_pickle=True).tolist()
        )
      with open(self._inv_payload_path, "rb") as f:
        self._inv_payload = pickle.load(f)
      try:
        assert self._index.ntotal == len(
          self.payload__
        ), "Number of Rows doesn't match"
      except:
        self._init = False
        raise
      return self
    print("No database found. Initializing a new one")
    self._init = False
    self._index = faiss.IndexFlatL2(self._vector_dim)
    self.payload__ = orderedset.OrderedSet()  # np.asarray([], dtype=str)
    self._inv_payload = dict()
    return self

  def keys(self):
    return self.payload__

  @property
  def _payload(self):
    if self.np_payload is None or id(self.payload__[-1]) != self.np_prev_hash:
      self.np_payload = np.asarray(list(self.payload__), dtype=str)
      self.np_prev_hash = id(self.payload__[-1])
    return self.np_payload

  def nearest(self, vector, num_closest=1):
    """
    Searches for the most similar items and returns them.
    Args:
     vector: a `np.float32` context vector of shape <= 2
     num_closest (default: 1): a `np.int32` variable
                  to denote the number of
                  closest matches to return
    Returns:
     A list of strings containing the similar items
    """
    if len(vector.shape) < 2:
      vector = vector[np.newaxis, :]
    _, index = self._index.search(vector, num_closest)
    payload_paths = self._payload[tuple(index)]
    return payload_paths.tolist()

  def insert(self, string, vector):
    """
    Inserts new item-vector pair to the database
    Args:
     string (str): The string item to store
     vector: a `np.float32` context vector for
         storing with the key
    Returns:
     `True` if insertion was successful `False` if
     insertion fails.
    """
    self._init = True
    if not string in self.payload__:
      if len(vector.shape) < 2:
        vector = vector[np.newaxis, :]
      self._index.add(vector)
      self.payload__.add(string)
      self._inv_payload[string] = len(self.payload__) - 1
      return True
    return False

  def search_vector(self, key):
    """
    Inverse Search for searching vectors for a given key.
    Args:
     key (str): key to be searched
    Returns:
     A `np.float32` context vector associated with the key
    """
    index = self._inv_payload.get(key, None)
    if index is not None:
      return self._index.reconstruct(index)
    return np.zeros(self._vector_dim, dtype=np.float32)

  def remove(self, index):
    """
    Removes items at a given index / indices.
    Args:
     index (int / list(int)): index / indices of the items to be removed
    Returns:
     `True` if the deletion was successful, `False` if not.
    """
    if self._payload.size:
      index = np.asarray(index)
      if index.ndim < 1:
        index = index[np.newaxis]
      keys = self._payload[index]
      self.payload__ = orderedset.OrderedSet(
        np.delete(self._payload, index).tolist()
      )
      self._index.remove_ids(index)
      for key in keys.tolist():
        self._inv_payload.pop(key)
      return True
    return False

  def remove_by_key(self, key):
    """
    Removes item by a given key
    Args:
     key (str): key to be removed
    Returns:
     `True` if deletion was successful else returns `False`
    """
    index = self._inv_payload.get(key, None)
    if index is not None:
      return self.remove(index)
    return False

  def write(self):
    """
    Commits the Changes to disk
    """
    faiss.write_index(self._index, self._index_file)
    with open(self._payload_path, "wb") as f:
      np.save(f, self._payload, allow_pickle=True)
    with open(self._inv_payload_path, "wb") as f:
      pickle.dump(self._inv_payload, f)
