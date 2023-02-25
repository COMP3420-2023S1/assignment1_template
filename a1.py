import numpy as np
from tensorflow.keras import models
from tensorflow.keras import layers

# Task 1 (2 marks)
def image_statistics(image):
     """Return a dictionary with the following statistics about the image. Assume that 
     the image is a colour image with three channels.
     - resolution: a tuple of the form (number_rows, number_columns).
     - centroid: a tuple of tree elements, one per channel, where each element 
          shows the average of the channel values of the corresponding pixel.
     - max_values: a tuple of three elements, one per channel, where each element
          shows the maximum value in each channel.
     - min_values: a tuple of three elements, one per channel, where each element 
          shows the minimum value in each channel.
     >>> image = np.array([[[250,   2,   2], [  0, 255,   2], [  0,   0, 255]], \
                           [[  2,   2,   2], [250, 255, 255], [127, 127, 127]]])                          
     >>> image_statistics(image) # doctest: +ELLIPSIS
     {'resolution': (2, 3, 3), 'centroid': (104.83..., 106.83..., 107.16...), 'max_values': (250, 255, 255), 'min_values': (0, 0, 2)}
     """

     return {}

# Task 2 (2 marks)
def brightness_mask(image, threshold):
     """Return a mask (that, is a Numpy array with only values 0 and 1) that indicates, 
     with value 1, all pixels that are brighter or equal than a threshold. To calculate the
     brightness of a pixel, compute the average of the channel values.
     >>> image = np.array([[[250,   2,   2], [  0, 255,   2], [  0,   0, 255]], \
                           [[  2,   2,   2], [250, 255, 255], [127, 127, 127]]])                          
     >>> brightness_mask(image, 120)
     array([[0, 0, 0],
            [0, 1, 1]])
     """

     return []

# Task 3 (2 marks)
def mask_togreyscale(image, mask):
     """Return a copy of the input colour image such that all pixels whose mask value is 1
     are converted as greyscale values. To convert a pixel to the greyscale values, change 
     each channel value to the mean value of the channels in the pixel.
     >>> image = np.array([[[250,   2,   2], [  0, 255,   2], [  0,   0, 255]], \
                           [[  2,   2,   2], [250, 255, 255], [127, 127, 127]]])
     >>> mask_togreyscale(image, np.array([[0,0,0],[0,1,1]]))
     array([[[250,   2,   2],
             [  0, 255,   2],
             [  0,   0, 255]],
     <BLANKLINE>
            [[  2,   2,   2],
             [253, 253, 253],
             [127, 127, 127]]])
     """

     return []

# Task 4 (2 marks)
def build_simple_nn(rows, columns, hidden_size, output_size, output_activation):
     """Return a Keras neural model model with the following layers:
     - A Flatten layer with input shape (rows, columns, 3)
     - A Dense layer with size hidden_size and activation 'relu'
     - An output layer with size output_size and activation output_activation
     >>> model = build_simple_nn(34, 56, 32, 6, 'softmax')
     >>> model.summary()
     Model: "sequential_1"
     _________________________________________________________________
      Layer (type)                Output Shape              Param #   
     =================================================================
      flatten_1 (Flatten)         (None, 5712)              0         
     <BLANKLINE>
      dense_3 (Dense)             (None, 32)                182816    
     <BLANKLINE>
      dense_4 (Dense)             (None, 6)                 198       
     <BLANKLINE>
     =================================================================
     Total params: 183,014
     Trainable params: 183,014
     Non-trainable params: 0
     _________________________________________________________________
     >>> model.get_layer(index=1).get_config()['activation']
     'relu'
     >>> model.get_layer(index=2).get_config()['activation']
     'softmax'
     """

     return None

# Task 5
def build_deep_nn(rows, columns, num_hidden, hidden_sizes, dropout_rates,
                  output_size, output_activation):
     """Return a Keras neural model that has the followink layers:
     - a Flatten layer with input shape (rows, columns, 3)
     - as many hidden layers as specified by num_hidden
       - hidden layer number i is of size hidden_sizes[i] and activation 'relu'
       - if dropout_rates[i] > 0, then hidden layer number 1 is followed
         by a dropout layer with dropout rate dropout_rates[i]
     - a final layer with size output_size and activation output_activation
     >>> model = build_deep_nn(45, 34, 2, (40, 20), (0, 0.5), 3, 'sigmoid')
     >>> model.summary()
     Model: "sequential"
     _________________________________________________________________
      Layer (type)                Output Shape              Param #   
     =================================================================
      flatten (Flatten)           (None, 4590)              0         
     <BLANKLINE>
      dense (Dense)               (None, 40)                183640    
     <BLANKLINE>
      dense_1 (Dense)             (None, 20)                820       
     <BLANKLINE>
      dropout (Dropout)           (None, 20)                0         
     <BLANKLINE>
      dense_2 (Dense)             (None, 3)                 63        
     <BLANKLINE>
     =================================================================
     Total params: 184,523
     Trainable params: 184,523
     Non-trainable params: 0
     _________________________________________________________________
     >>> model.get_layer(index=1).get_config()['activation']
     'relu'
     >>> model.get_layer(index=2).get_config()['activation']
     'relu'
     >>> model.get_layer(index=4).get_config()['activation']
     'sigmoid'

     """
     
     return None

if __name__ == "__main__":
     import doctest
     doctest.testmod()