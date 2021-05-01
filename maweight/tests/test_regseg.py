from unittest import TestCase

import maweight as rs
import numpy as np
import nibabel as nib

class Testmaweight(TestCase):
    def test_version_check(self):
        self.assertTrue('elastix' in rs.executable_version())
    
    def test_on_ndarrays(self):
        s= 17
        a= np.zeros((s, s, s))
        a[5, 5, 5]= 1
        a[5, 4, 5]= 1
        a[5, 6, 5]= 1
        a[4, 5, 5]= 1
        a[6, 5, 5]= 1
        a[5, 5, 4]= 1
        a[5, 5, 6]= 1
        b= np.zeros((s, s, s))
        b[5, 4, 3]= 2
        b[4, 4, 3]= 2
        b[6, 4, 3]= 2
        b[5, 4, 3]= 2
        b[5, 6, 3]= 2
        b[5, 4, 4]= 2
        b[5, 6, 6]= 2
        
        c= np.zeros((s, s, s))
        c[5, 5, 5]= 10
        d= np.zeros((s, s, s))
        d[5, 4, 5]= 9
        
        results= rs.register_and_transform(a, b, [c, d], None,
                                        work_dir=None, verbose= 0)
        self.assertTrue(isinstance(results[0], nib.Nifti1Image))
