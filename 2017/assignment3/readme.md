# Note

For the TensorFlow version of the Style Transfer example, **DO NOT** use `tf.norm` for the first question. **INSTEAD USE** `tf.reduce_sum` .

Although you will get the right result for that question, the pictures you will generate at the end will be wrong (the values for the pixels of the images will be NaNs). It might have something to do with `tf.norm` taking the square root (before being squared again in our case).

See https://github.com/tensorflow/tensorflow/issues/12071 for a discussion of the problem.
