from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.INFO)
#^^^^^ is all as configured in the example

def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""
  # Input Layer
  input_layer = tf.reshape(features["x"], [-1, 28, 28, 1]) # -1 means all the samples we feed it. 28's mean 28x28 pixel image, so change to what we need. 1 is for number of channels, 1 for monochrome, 3 for rgb

  # Convolutional Layer #1
  conv1 = tf.layers.conv2d(
      inputs=input_layer, # input tensor, leave this be once we fix input layer.
      filters=32, # Gives the dimensionality of output tensor
      kernel_size=[5, 5], # size of the convolutional window, so the picture is looked at in 5x5 windows. We can change to whatever.
      padding="same", # Specifies that the output tensor should have same width and height at input, so it pads with 0's if need be so everything works out. I think we want this here.
      activation=tf.nn.relu) # activation function, currently ReLU but we can make it whatever*, so we can discuss. Maybe sigmoid? Not sure of relative benefits.

      # With values as-is, the output tensor is [-1 (because of batch size), 28, 28 (because of 'same'), 32 (because of filters)]

  # Pooling Layer #1
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
      # uses max pooling, where it selects the max values from the neuron clusters from the last layer. We could also use average.
      # takes previous output as input.
      # pool size specifies which pools you'll draw the max values from
      # strides specifies the separation of the pools, in this case it means 2 pixels.
      # In this case, ouputs a [-1, 14, 14, 32] tensor since the 2x2 pools half the height and width of the input.

  # Convolutional Layer #2 and Pooling Layer #2
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

      # outputs [-1, 14, 14, 64]

  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
      # outputs [-1, 7, 7, 64]

  # Dense Layer
  pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
  # Reshapes the tensor into a vector which will feed into the dense layer

  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
  # takes the flat tensor, specifies 1024 neurons, and uses ReLU as the activation function

  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
    # Randomly drops 40% of elements during training. This is done for logical reasons, I'm sure.
    # has shape [-1, 1024]

  # Logits Layer
  logits = tf.layers.dense(inputs=dropout, units=10)
  # This creates a dense layer with a neuron for each output class, so we'd need one for each author, so like 50 instead of 10.
  # outputs a tensor of size [-1, 10].

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # This returns the highest value in the output tensor, corresponding to a prediction of the most likely author for us.

      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
      # This generates a tensor of probabilities that each entry is correct.
  }

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    # returns the predictions object, both the best guess and the probabilities.

  # Calculate Loss (for both TRAIN and EVAL modes) using cross entropy. We can switch the loss calculation method if we want. This doesn't line up with the explaination in the article, but maybe the explaination was for outdated code.
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  # Use stochastic gradient descent with learning rate 1 to train, minimizing loss.
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def main(unused_argv):

  # This loads in the training and evaluation data. We might need to change this depending on the Part1 stuff, but here it is as in the example.
  #mnist = tf.contrib.learn.datasets.load_dataset("mnist")
  #print(type(mnist))


  sess = tf.Session()
  filenames = tf.placeholder(tf.string, shape=[None])
  dataset   = tf.data.TFRecordDataset(filenames)
  #dataset   = dataset.map(...)  # Parse the record into tensors.
  #dataset   = dataset.repeat()  # Repeat the input indefinitely.
  #dataset   = dataset.batch(32)
  iterator  = dataset.make_initializable_iterator()
  print(type(dataset))
  # You can feed the initializer with the appropriate filenames for the current
  # phase of execution, e.g. training vs. validation.
  #exit()
  # Initialize `iterator` with training data.
  training_filenames = ["./output/train-00-of-04.tfrecord", "./output/train-01-of-04.tfrecord",
                        "./output/train-02-of-04.tfrecord", "./output/train-03-of-04.tfrecord"]
  sess.run(iterator.initializer, feed_dict={filenames: training_filenames})

  # Initialize `iterator` with validation data.
  validation_filenames = ["./output/validation-00-of-04.tfrecord", "./output/validation-01-of-04.tfrecord",
                          "./output/validation-02-of-04.tfrecord", "./output/validation-03-of-04.tfrecord"]
  sess.run(iterator.initializer, feed_dict={filenames: validation_filenames})

  train_data = dataset.train.images # mnist.train.images
  train_labels = np.asarray(dataset.train.labels, dtype=np.int32) # mnist.train.labels, dtype=np.int32)
  eval_data = dataset.test.images   # mnist.test.images
  eval_labels = np.asarray(dataset.test.labels, dtype=np.int32)   # mnist.test.labels, dtype=np.int32)

  # The estimator definition, which is what uses the large function above.
  classifier = tf.estimator.Estimator(model_fn= cnn_model_fn, model_dir="/wherever/we/want/to/keep/temp/data")

  # Set up logging for while the thing is running
  tensors_to_log = {"probabilities": "softmax_tensor"}
  logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50) # log ever 50 steps in training, can be changed

  # Training fucntion to do the actual training
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={ "x" :train_data}, # feature data is passed here as a dictionary
    y=train_labels, # labels are passed here, so that would be our authors
    batch_size=100, # we train on batches of 100 samples at a time, can be changed if desired
    num_epochs=None, # No epochs, don't know if we want them
    shuffle=True) # Shuffle the data around as we train for better results

  classifier.train(
    input_fn=train_input_fn,
    steps=2000, # train for 20,000 steps total, can change if we want
    hooks=[logging_hook])

  # This evaluates the training and prints results
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x":eval_data},
      y=eval_labels,
      num_epochs=1, # Iterates over one epoch of data, dunno what that means
      shuffle=False) # Does not shuffle the data as we try and evaluate (probably for preformance)

  #eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
  #print(eval_results)



if __name__ == "__main__":
  tf.app.run()

