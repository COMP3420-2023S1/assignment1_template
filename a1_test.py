import numpy as np
import tensorflow as tf
import unittest

import a1

def get_model_summary(model: tf.keras.Model) -> str:
    "Adapted from https://stackoverflow.com/questions/41665799/keras-model-summary-object-to-string"
    string_list = []
    model.summary(line_length=80, print_fn=lambda x: string_list.append(x))
    return "\n".join(string_list)

class TestBasic(unittest.TestCase):
    def test_q1(self):
        image = np.array([[[250,   2,   2], [  0, 255,   2], [  0,   0, 255]], \
                          [[  2,   2,   2], [250, 255, 255], [127, 127, 127]]])                          
        target = {'resolution': (2, 3, 3), 
                  'centroid': (104.833, 106.833, 107.167), 
                  'max_values': (250, 255, 255), 
                  'min_values': (0, 0, 2)}
        result = a1.image_statistics(image)
        self.assertEqual(result['resolution'], target['resolution'])
        for i in range(3):
            self.assertAlmostEqual(result['centroid'][i], 
                                   target['centroid'][i], 
                                   places=3)
        self.assertEqual(result['max_values'], target['max_values'])
        self.assertEqual(result['min_values'], target['min_values'])

    def test_q2(self):
        image = np.array([[[250,   2,   2], [  0, 255,   2], [  0,   0, 255]], \
                          [[  2,   2,   2], [250, 255, 255], [127, 127, 127]]])                          
        target = np.array([[0, 0, 0], [0, 1, 1]])
        result = a1.brightness_mask(image, 120)
        np.testing.assert_array_equal(result, target)

    def test_q3(self):
        image = np.array([[[250,   2,   2], [  0, 255,   2], [  0,   0, 255]], \
                          [[  2,   2,   2], [250, 255, 255], [127, 127, 127]]])                          
        target = np.array([[[250,   2,   2],
                            [  0, 255,   2],
                            [  0,   0, 255]],
                           [[  2,   2,   2],
                            [253, 253, 253],
                            [127, 127, 127]]])
        result = a1.mask_togreyscale(image, np.array([[0,0,0],[0,1,1]]))
        np.testing.assert_array_equal(result, target)

    def test_q4(self):
        self.maxDiff = None
        model = a1.build_simple_nn(34, 56, 32, 6, 'softmax')
        target = """Model: "sequential"
________________________________________________________________________________
 Layer (type)                       Output Shape                    Param #     
================================================================================
 flatten (Flatten)                  (None, 5712)                    0           
                                                                                
 dense (Dense)                      (None, 32)                      182816      
                                                                                
 dense_1 (Dense)                    (None, 6)                       198         
                                                                                
================================================================================
Total params: 183,014
Trainable params: 183,014
Non-trainable params: 0
________________________________________________________________________________"""
        self.assertEqual(get_model_summary(model), target)
        self.assertEqual(model.get_layer(index=1).get_config()['activation'],'relu')
        self.assertEqual(model.get_layer(index=2).get_config()['activation'],'softmax')

    def test_q5(self):
        self.maxDiff = None
        model = a1.build_deep_nn(45, 34, 2, (40, 20), (0, 0.5), 3, 'sigmoid')
        target = """Model: "sequential"
________________________________________________________________________________
 Layer (type)                       Output Shape                    Param #     
================================================================================
 flatten (Flatten)                  (None, 4590)                    0           
                                                                                
 dense (Dense)                      (None, 40)                      183640      
                                                                                
 dense_1 (Dense)                    (None, 20)                      820         
                                                                                
 dropout (Dropout)                  (None, 20)                      0           
                                                                                
 dense_2 (Dense)                    (None, 3)                       63          
                                                                                
================================================================================
Total params: 184,523
Trainable params: 184,523
Non-trainable params: 0
________________________________________________________________________________"""
        self.assertEqual(get_model_summary(model), target)
        self.assertEqual(model.get_layer(index=1).get_config()['activation'],'relu')
        self.assertEqual(model.get_layer(index=2).get_config()['activation'],'relu')
        self.assertEqual(model.get_layer(index=4).get_config()['activation'],'sigmoid')
