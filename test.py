import tensorflow as tf
import tensorflow_hub as hub

# Download manually (only needs to be done once with internet):
hub_model_url = "https://tfhub.dev/google/yamnet/1"
local_model_path = "./yamnet_model"  # folder to store

# This will download and save it locally
yamnet_model = hub.load(hub_model_url)
tf.saved_model.save(yamnet_model, local_model_path)

# Later, even offline, you can load from disk:
yamnet_model = hub.load(local_model_path)

