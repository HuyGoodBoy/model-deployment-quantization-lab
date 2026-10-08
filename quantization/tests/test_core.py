import unittest

import numpy as np

from qlab.common import classification_metrics, dequantize_tensor, numerical_metrics, quantize_tensor
from qlab.prepare import balanced_selection, normalize_text


class QuantizationMathTests(unittest.TestCase):
    def test_clip_before_int8_cast_prevents_wraparound(self):
        values=np.array([-1000,-1,0,1,1000],dtype=np.float32)
        q=quantize_tensor(values,np.int8,0.1,-3)
        np.testing.assert_array_equal(q,[-128,-13,-3,7,127])

    def test_half_step_round_trip_error_bound(self):
        values=np.array([-0.35,-0.07,0.09,0.36],dtype=np.float32)
        restored=dequantize_tensor(quantize_tensor(values,np.int8,0.02,4),0.02,4)
        self.assertLessEqual(float(np.abs(values-restored).max()),0.010001)

    def test_unsigned_zero_point_without_underflow(self):
        restored=dequantize_tensor(np.array([0,128,255],dtype=np.uint8),0.5,128)
        np.testing.assert_array_equal(restored,[-64,0,63.5])

    def test_invalid_scale_is_rejected(self):
        for scale in (0,-1,np.nan):
            with self.assertRaises(ValueError): quantize_tensor(np.array([0]),np.int8,scale,0)


class LabelAndMetricTests(unittest.TestCase):
    def test_binary_f1_uses_ground_truth(self):
        logits=np.array([[2,0],[0,2],[0,1],[0,1]],dtype=np.float32)
        labels=np.array([0,1,0,1])
        result=classification_metrics(logits,labels,'text')
        self.assertEqual(result['accuracy'],0.75)
        self.assertAlmostEqual(result['f1_positive'],0.8)
        self.assertNotIn('top5_accuracy',result)

    def test_1000_class_prediction_is_not_masked_to_dataset_classes(self):
        output=np.zeros((1,1000),dtype=np.float32)
        output[0,900]=1
        result=classification_metrics(output,np.array([50]),'cv')
        self.assertEqual(result['accuracy'],0)
        self.assertEqual(result['top5_accuracy'],0)

    def test_selection_balanced_and_reproducible(self):
        records=list(range(30));labels=[i//10 for i in records]
        a=balanced_selection(records,labels,3)
        self.assertEqual(a,balanced_selection(records,labels,3))
        self.assertEqual([sum(labels[x]==k for x in a) for k in range(3)],[3,3,3])
        self.assertEqual(len(set(a)),9)
        self.assertEqual(normalize_text('  A  GREAT\n Movie '),'a great movie')

    def test_snr_power_ratio_and_nonfinite_rejection(self):
        a=np.ones((1,4),dtype=np.float32);b=a+np.float32(0.1)
        self.assertAlmostEqual(numerical_metrics(a,b)['snr_db'],20,places=4)
        with self.assertRaises(ValueError): numerical_metrics(a,b*np.nan)


if __name__=='__main__': unittest.main()
