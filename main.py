from __future__  import absolute_import, division, print_function, unicode_literals
import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers
from typing import Dict
import numpy as np 
import tensorflow as tf
from tensorflow.keras import layers, models
from typing import List, Dict, Tuple
import os
import spacy
import json
nlp = spacy.load("en_core_web_sm", disable = ['ner', 'tagger', 'parser', 'textcat'])
from collections import Counter
from typing import List, Dict, Tuple, Any
import json
import os
import zipfile

# external lib imports:
import numpy as np
from tqdm import tqdm
import spacy

from typing import List, Dict
import os
import argparse
import random
import json

# inbuilt lib imports:
from typing import List, Dict
import os
import argparse
import random
import json

# external lib imports:
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from tensorflow.keras import models, optimizers


# external lib imports:
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from tensorflow.keras import models, optimizers

# read the common words
def read_common_words():
  words = []
  with open('glove_common_words.txt', 'r') as open_file:
    for line in open_file:
      words.append(line.strip())
  return words


def read_instances(file_path, max_tokens):
  responses = []
  with open(file_path, 'r') as open_file:
    for line in open_file:
      line = line.strip()
      instance = json.loads(line)
      text = instance['text']
      tokens = [token.text.lower() for token in nlp.tokenizer(text)][:max_tokens]
      response = dict()
      instance['text_tokens'] = tokens
      instance['labels'] = instance.pop("label", None)
      instance.pop('text')
      responses.append(instance)
  return responses

def build_vocabulary(instances, vocab_size = 10000, add_tokens = None):
  UNK_TOKEN = "@UNK@"
  PAD_TOKEN = "@PAD@"
  token_to_id = {PAD_TOKEN: 0, UNK_TOKEN: 1}

  # First add tokens which were explicitly passed.
  add_tokens = add_tokens or []
  for token in add_tokens:
      if not token.lower() in token_to_id:
          token_to_id[token] = len(token_to_id)

  # Add remaining tokens from the instances as the space permits
  words = []
  for instance in instances:
      words.extend(instance["text_tokens"])
  token_counts = dict(Counter(words).most_common(vocab_size))
  for token, _ in token_counts.items():
      if token not in token_to_id:
          token_to_id[token] = len(token_to_id)
      if len(token_to_id) == vocab_size:
          break
  # Make reverse vocabulary lookup
  id_to_token = dict(zip(token_to_id.values(), token_to_id.keys()))
  return (token_to_id, id_to_token)

  token_to_id, id_to_token = build_vocabulary(example_instances, 10000, common_words)

  def index_instances(instances, token_to_id):
    for instance in instances:
        token_ids = []
        for token in instance['text_tokens']:
            if token in token_to_id:
                token_ids.append(token_to_id[token])
            else:
                token_ids.append(0)
            instance['text_token_ids'] = token_ids
    return instances

def split_dicts(instances):
  response = []
  for instance in instances:
    response.append((instance['text_token_ids'], instance['labels']))
  return response

def load_glove_embeddings(embeddings_txt_file: str,
                          embedding_dim: int,
                          vocab_id_to_token: Dict[int, str]) -> np.ndarray:
    """
    Given a vocabulary (mapping from index to token), this function builds
    an embedding matrix of vocabulary size in which ith row vector is an
    entry from pretrained embeddings (loaded from embeddings_txt_file).
    """
    tokens_to_keep = set(vocab_id_to_token.values())
    vocab_size = len(vocab_id_to_token)

    embeddings = {}
    print("\nReading pretrained embedding file.")
    with open(embeddings_txt_file, encoding='utf8') as file:
        for line in tqdm(file):
            line = str(line).strip()
            token = line.split(' ', 1)[0]
            if not token in tokens_to_keep:
                continue
            fields = line.rstrip().split(' ')
            if len(fields) - 1 != embedding_dim:
                raise Exception(f"Pretrained embedding vector and expected "
                                f"embedding_dim do not match for {token}.")
                continue
            vector = np.asarray(fields[1:], dtype='float32')
            embeddings[token] = vector

    # Estimate mean and std variation in embeddings and initialize it random normally with it
    all_embeddings = np.asarray(list(embeddings.values()))
    embeddings_mean = float(np.mean(all_embeddings))
    embeddings_std = float(np.std(all_embeddings))

    embedding_matrix = np.random.normal(embeddings_mean, embeddings_std,
                                        (vocab_size, embedding_dim))
    embedding_matrix = np.asarray(embedding_matrix, dtype='float32')

    for idx, token in vocab_id_to_token.items():
        if token in embeddings:
            embedding_matrix[idx] = embeddings[token]

    return embedding_matrix

def generate_batches(instances: List[Dict], batch_size) -> List[Dict[str, np.ndarray]]:
    """
    Generates and returns batch of tensorized instances in a chunk of batch_size.
    """

    def chunk(items: List[Any], num: int):
        return [items[index:index+num] for index in range(0, len(items), num)]
    batches_of_instances = chunk(instances, batch_size)

    batches = []
    for batch_of_instances in tqdm(batches_of_instances):

        num_token_ids = [len(instance["text_tokens_ids"])
                         for instance in batch_of_instances]
        max_num_token_ids = max(num_token_ids)

        count = min(batch_size, len(batch_of_instances))
        batch = {"inputs": np.zeros((count, max_num_token_ids), dtype=np.int32)}
        if "labels" in  batch_of_instances[0]:
            batch["labels"] = np.zeros(count, dtype=np.int32)

        for batch_index, instance in enumerate(batch_of_instances):
            num_tokens = len(instance["text_tokens_ids"])
            inputs = np.array(instance["text_tokens_ids"])
            batch["inputs"][batch_index][:num_tokens] = inputs

            if "labels" in instance:
                batch["labels"][batch_index] = np.array(instance["labels"])
        batches.append(batch)

    return batches

class MainModel(tf.keras.Models):
  def __init__(self, vocab_size, embedding_dim, num_classes= 2, num_layers = 3):
    super(MainModel, self).__init__(self)
    self._vocab_size = _vocab_size
    self._embedding_dim = embedding_dim
    self._num_classes = num_classes
    self._num_layers = num_layers
    self._embeddings = tf.Variable(tf.random_normal((vocab_size, embedding_dim)))
    self._layers = [keras.Dense(1000, activation='relu =') for i in range(num_layers)]
    self._classfication_layer = keras.Dense(units=num_classes)
  
  def call(self, inputs, training = False):
    embedding_tokens = tf.nn.embedding_lookup(self._embeddings, inputs)
    tokens_mask = tf.cast(inputs!=0, tf.float32)
    resp = embedding_tokens
    for i in range(self._num_layers):
      resp = self._layers[i](resp)
    return self._classification_layer(resp)


def train(model, optimizer, train_instances, validation_instances, num_epochs = 5, batch_size = 250):
    train_batches = generate_batches(train_instances, batch_size)
    validation_batches = generate_batches(validation_instances, batch_size)

    train_batch_labels = [batch_inputs.pop("labels") for batch_inputs in train_batches]
    validation_batch_labels = [batch_inputs.pop("labels") for batch_inputs in validation_batches]

    for epoch in range(num_epochs):
        total_training_loss = 0
        total_correct_predictions, total_predictions = 0, 0
        generator_tqdm = tqdm(list(zip(train_batches, train_batch_labels)))
        for index, (batch_inputs, batch_labels) in enumerate(generator_tqdm):
            with tf.GradientTape() as tape:
                logits = model(**batch_inputs)
                loss_value = cross_entropy_loss(logits, batch_labels)
                grads = tape.gradient(loss_value, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            total_training_loss += loss_value
            batch_predictions = np.argmax(tf.nn.softmax(logits, axis=-1).numpy(), axis=-1)
            total_correct_predictions += (batch_predictions == batch_labels).sum()
            total_predictions += batch_labels.shape[0]
            description = ("Average training loss: %.2f Accuracy: %.2f "
                        % (total_training_loss/(index+1), total_correct_predictions/total_predictions))
            generator_tqdm.set_description(description, refresh=False)
        average_training_loss = total_training_loss / len(train_batches)
        training_accuracy = total_correct_predictions/total_predictions


def main():

    train_instances = read_instances('imdb_sentiment_train_15k.jsonl', 250)
    val_instances = read_instances('imdb_sentiment_dev.jsonl' ,250)

    common_words = []
    with open(GLOVE_COMMON_WORDS_PATH, encoding='utf8') as file:
        common_words = [line.strip() for line in file.readlines() if line.strip()]
    
    token_to_id, id_to_token = build_vocabulary(train_instances, 10000, common_words)

    train_instances = index_instances(train_instances, token_to_id)
    val_instances = index_instances(val_instances, token_to_id)

    embeddings = load_glove_embeddings('glove.6B.300d.txt', 300, id_to_token)

    mainModel = MainModel(len(vocab_token_to_id), 300, num_classes= 2, num_layers = 3)
    
    optimizer = optimizers.Adam()

    train(mainModel, optimizer, train_instances, validation_instances, num_epochs = 5, batch_size = 250)

if __name__ == '__main__':
    main()

# train data labels

# create model

# set loss function and optimizer

# fit the model with data

# evaluate on the results of the data