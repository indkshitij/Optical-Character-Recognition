import tensorflow as tf
import keras
import numpy as np
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers.experimental.preprocessing import StringLookup

batch_size = 32
padding_token = 99
image_width = 128
image_height = 32
image_path = r"C:\Users\indks\Desktop\F\images\scripts.jpg"
max_len=32
vocab1 = ['b', 'k', 'T', 'n', 'I', 'p', '-', '1', 'h', 'N', '8', '(', ';', 'M', 'B', 'W', 'S', '3', 'a', 'y', 'F', 'Z', '6', '4', 'j', 'K', 'e', '/', ',', 'r', 's', 'l', 't', 'x', 'q', '9', '2', '*', 'L', '"', 'w', 'A', '7', 'z', 'C', 'J', 'E', 'G', '#', 'f', 'R', 'u', '5', '!', '.', 'i', 'X', 'd', 'Q', 'P', 'Y', 'V', 'm', "'", 'O', 'o', 'c', 'v', 'U', 'D', '0', 'g', ')', '?', ':', 'H']


class CTCLayer(keras.layers.Layer):
    def __init__(self, name=None, **kwargs):
        # We don't need to handle 'trainable' explicitly, just use **kwargs to pass other args
        super(CTCLayer, self).__init__(name=name, **kwargs)
        self.loss_fn = tf.keras.backend.ctc_batch_cost  # CTC loss function from Keras backend

    def call(self, y_true, y_pred):
        # Compute the batch size
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        
        # Compute the input sequence length (number of time steps)
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        
        # Compute the label sequence length
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        # Replicate the lengths for each item in the batch
        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

        # Compute the CTC loss for the batch
        loss = self.loss_fn(y_true, y_pred, input_length, label_length)

        # Add the loss to the model (for tracking)
        self.add_loss(loss)

        # Return the predictions (logits) at inference time
        return y_pred

def distortion_free_resize(image, img_size):
    w, h = img_size
    image = tf.image.resize(image, size=(h, w), preserve_aspect_ratio=True)

    # Check tha amount of padding needed to be done.
    pad_height = h - tf.shape(image)[0]
    pad_width = w - tf.shape(image)[1]

    # Only necessary if you want to do same amount of padding on both sides.
    if pad_height % 2 != 0:
        height = pad_height // 2
        pad_height_top = height + 1
        pad_height_bottom = height
    else:
        pad_height_top = pad_height_bottom = pad_height // 2

    if pad_width % 2 != 0:
        width = pad_width // 2
        pad_width_left = width + 1
        pad_width_right = width
    else:
        pad_width_left = pad_width_right = pad_width // 2

    image = tf.pad(
        image,
        paddings=[
            [pad_height_top, pad_height_bottom],
            [pad_width_left, pad_width_right],
            [0, 0],
        ],
    )

    image = tf.transpose(image, perm=[1, 0, 2])
    image = tf.image.flip_left_right(image)
    return image

def preprocess_image(image_path, img_size=(image_width, image_height)):

    # Load the image
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=1)  # Decode PNG as grayscale
    
    # Resize the image with aspect ratio preserved and padding
    image = distortion_free_resize(image, img_size)
    
    # Normalize pixel values to [0, 1]
    image = tf.cast(image, tf.float32) / 255.0
    
    # Add batch dimension to match model input
    image = tf.expand_dims(image, axis=0)  # Shape: [1, height, width, channels]
    
    return image

def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search.
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][
        :, :max_len
    ]
    # Iterate over the results and get back the text.
    output_text = []
    for res in results:
        res = tf.gather(res, tf.where(tf.math.not_equal(res, -1)))
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
        output_text.append(res)
    return output_text

def predict_text(image_path):
    # Preprocess the image
    img = preprocess_image(image_path,img_size=(image_width, image_height))
    dummy_labels = tf.zeros((1,128), dtype=tf.int32)

    # Get predictions from the model
    predictions = model.predict([img,dummy_labels])

    
    # # Decode predictions
    decoded_prediction = decode_batch_predictions(predictions)
    
    return decoded_prediction


model_path = r"C:\Users\indks\Desktop\F\path_to_model.h5"


model = tf.keras.models.load_model(model_path,custom_objects={"CTCLayer": CTCLayer})


AUTOTUNE = tf.data.AUTOTUNE

# Mapping characters to integers.
char_to_num = StringLookup(vocabulary=vocab1, mask_token=None)

# Mapping integers back to original characters.
num_to_char = StringLookup(vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True)


predicted=predict_text(image_path)
print(predicted)

