import tensorflow as tf

class CTCLayer(tf.keras.layers.Layer):
    def __init__(self, name = None):
        super().__init__(name = name)
        self.loss_fn = tf.keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):
        # Number of samples in a batch.
        batch_len = tf.cast(tf.shape(y_true)[0], dtype = "int64")
        # Length of the "time steps".
        input_length = tf.cast(tf.shape(y_pred)[1], dtype = "int64")
        input_length = input_length * tf.ones(shape = (batch_len, 
                                             1), dtype = "int64")
        # Length of the label.
        label_length = tf.cast(tf.shape(y_true)[1], dtype = "int64")
        label_length = label_length * tf.ones(shape = (batch_len, 
                                             1), dtype = "int64")
        loss = self.loss_fn(y_true, y_pred, 
                input_length, label_length)
        self.add_loss(loss)
        return y_pred
