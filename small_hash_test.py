import tensorflow as tf

from testw2v import google_example, config, gensim_utils
from testw2v.hash_embedding import hash_embedding, hash_embedding_verbose
tf.config.run_functions_eagerly(False)
global_conf = config.load()
conf = global_conf['experiments']['google_default']
conf = {'batch_size': 64,
 'buffer_size': 10000,
 'embedding_dim': 128,
 'epochs': 20,
 'num_ns': 10,
 'seed': 42,
 'sequence_length': 100,
 'window_size': 4,
 'vocab_size': 4096}
dataset, vector_layer = google_example.file_to_dataset('wiki_1pct', conf)

class Word2Vec(tf.keras.Model):
  def __init__(self, embedding_width, num_words=1024, num_hash_buckets=2**20, num_hash_func=3):
    super(Word2Vec, self).__init__()

    self.target_embedding = hash_embedding.HashEmbedding(num_hash_func=num_hash_func, 
            num_words=num_words, 
            num_hash_buckets=num_hash_buckets, # ~1MM 
            embedding_width=embedding_width, 
            random_seed=1139, 
            name="w2v_target_embedding") 
 
    self.context_embedding = hash_embedding.HashEmbedding(num_hash_func=num_hash_func, 
            num_words=num_words, 
            num_hash_buckets=num_hash_buckets, # ~1MM 
            embedding_width=embedding_width, 
            random_seed=1139, 
            name="w2v_context_embedding") 
 
    self.dots = tf.keras.layers.Dot(axes=(3,2),dtype=tf.float32) 
    self.flatten = tf.keras.layers.Flatten() 
 
 
  def call(self, pair): 
    target, context = pair 
    we = self.target_embedding(target) 
    ce = self.context_embedding(context) 
    dots = self.dots([ce, we]) 
    return self.flatten(dots)

def build_model():
    embedding_dim = conf['embedding_dim'] 
    word2vec = Word2Vec(num_words=conf['vocab_size'], num_hash_buckets=2**17, embedding_width=embedding_dim)
    word2vec.compile(optimizer='adam',
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    return word2vec

model = build_model()

import datetime 

logs = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

tboard_callback = tf.keras.callbacks.TensorBoard(log_dir = logs,
                                                 histogram_freq = 1,
                                                 profile_batch = '25,35',
                                                 embeddings_freq=1
                                                 )


model.fit(dataset, epochs=10, callbacks=[tboard_callback])

vocab = vector_layer.get_vocabulary()
emb = model.layers[0](tf.constant([range(len(vocab))]))
weights = tf.squeeze(emb).numpy()
gensim_obj = gensim_utils.w2v_to_gensim(vocab, weights)
google_metrics = gensim_utils.metrics(gensim_obj)
print(google_metrics)