# Code style: AI-code. AI-generated material-map projection and PDF checks.
import importlib.util
import math
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import numpy as np


spec = importlib.util.spec_from_file_location('material_map',
    Path(__file__).parents[1]/'utils'/'plot_shift_material_map.py')
material_map = importlib.util.module_from_spec(spec)
spec.loader.exec_module(material_map)


class MaterialMapPlotTest(unittest.TestCase):
    def test_polar_theta_is_from_minus_z_and_phi_agrees_with_cms_xy(self):
        grid = dict(origin_cm=[0.,0.,300.],target_z_cm=200.,
                    eta_edges=[-math.asinh(2.),-math.asinh(1.)],
                    phi_edges=[0.,math.pi/2.,math.pi])
        azimuth,theta = material_map.polar_coordinates(grid)
        np.testing.assert_allclose(theta[0],[math.degrees(math.atan2(1.,2.)),45.],atol=1.e-13)
        np.testing.assert_allclose(azimuth[:,0],[0.,math.pi/2.,math.pi],atol=1.e-14)
        x,y = material_map.xy_coordinates(grid)
        # The endpoint is 1 m downstream: r=tan(theta) m, with CMS phi unchanged.
        np.testing.assert_allclose(x,np.tan(np.radians(theta))*np.cos(azimuth),atol=1.e-14)
        np.testing.assert_allclose(y,np.tan(np.radians(theta))*np.sin(azimuth),atol=1.e-14)

    def test_polar_does_not_invent_a_ray_at_zero_theta(self):
        grid = dict(origin_cm=[0.,0.,14800.],target_z_cm=1100.,
                    eta_edges=[-9.,-3.],phi_edges=[-math.pi,math.pi])
        _,theta = material_map.polar_coordinates(grid)
        self.assertAlmostEqual(theta[0,0],.014142,places=6)
        self.assertAlmostEqual(theta[0,-1],5.700470,places=5)
        self.assertGreater(theta.min(),0.)
        grid['origin_cm'][0] = 1.
        with self.assertRaisesRegex(ValueError,'on-axis source'):
            material_map.polar_coordinates(grid)

    def test_endpoint_projection_uses_cms_axes_and_converts_cm_to_metres(self):
        grid = dict(origin_cm=[100.,200.,300.],target_z_cm=200.,
                    eta_edges=[-math.asinh(2.),-math.asinh(1.)],
                    phi_edges=[0.,math.pi/2.,math.pi])
        x,y = material_map.xy_coordinates(grid)
        # A 100 cm flight in z with pz/pT=-2 or -1 gives 50 or 100 cm in r.
        np.testing.assert_allclose(x,[[1.5,2.],[1.,1.],[.5,0.]],atol=1.e-14)
        np.testing.assert_allclose(y,[[2.,2.],[2.5,3.],[2.,2.]],atol=1.e-14)
        grid.update(origin_cm=[100.,200.,-300.],target_z_cm=-200.,
                    eta_edges=[math.asinh(1.),math.asinh(2.)])
        mirrored_x,mirrored_y = material_map.xy_coordinates(grid)
        np.testing.assert_allclose(mirrored_x,x[:,::-1],atol=1.e-14)
        np.testing.assert_allclose(mirrored_y,y[:,::-1],atol=1.e-14)

    def test_no_projection_through_a_transverse_or_outward_ray(self):
        grid = dict(origin_cm=[0.,0.,14800.],target_z_cm=1100.,
                    eta_edges=[-3.,0.],phi_edges=[-math.pi,math.pi])
        with self.assertRaises(ValueError):
            material_map.xy_coordinates(grid)
        grid['eta_edges'] = [1.,2.]
        with self.assertRaises(ValueError):
            material_map.xy_coordinates(grid)

    def test_radiation_xy_pdf_is_readable_and_retains_scope_labels(self):
        data = dict(angular_grid=dict(origin_cm=[0.,0.,14800.],target_z_cm=1100.,
            eta_edges=[-9.,-6.,-3.],phi_edges=[-math.pi,-math.pi/2.,0.,math.pi/2.,math.pi]),paths=[])
        for i in range(2):
            for j in range(4):
                data['paths'].append(dict(eta_index=i,phi_index=j,complete=(i,j)!=(1,2),
                                          radiation_depth_proxy=200.+400.*i+200.*j))
        with tempfile.TemporaryDirectory(prefix='shift-material-map-test-') as directory:
            path = Path(directory)/'map.pdf'
            material_map.plot(data,path,view='xy',quantity='radiation')
            self.assertTrue(path.read_bytes().startswith(b'%PDF-'))
            self.assertGreater(path.stat().st_size,1000)
            if shutil.which('pdftotext'):
                text = subprocess.check_output(['pdftotext',str(path),'-'],text=True)
                self.assertIn('collimator z=148 m',text)
                self.assertIn('CMS entrance z=11 m',text)
                self.assertIn('Gray: untraced directions',text)
                self.assertIn('Integrated radiation lengths',text)

    def test_polar_radiation_pdf_retains_direction_and_material_units(self):
        data = dict(angular_grid=dict(origin_cm=[0.,0.,14800.],target_z_cm=1100.,
            eta_edges=[-9.,-6.,-3.],phi_edges=[-math.pi,-math.pi/2.,0.,math.pi/2.,math.pi]),paths=[])
        for i in range(2):
            for j in range(4):
                data['paths'].append(dict(eta_index=i,phi_index=j,complete=(i,j)!=(0,1),
                                          radiation_depth_proxy=200.+400.*i+200.*j))
        with tempfile.TemporaryDirectory(prefix='shift-material-polar-test-') as directory:
            path = Path(directory)/'polar.pdf'
            material_map.plot(data,path,view='theta-phi',quantity='radiation')
            self.assertTrue(path.read_bytes().startswith(b'%PDF-'))
            if shutil.which('pdftotext'):
                text = subprocess.check_output(['pdftotext',str(path),'-'],text=True)
                self.assertIn('Radius:',text)
                self.assertIn('from line to CMS [degrees]',text)
                self.assertIn('Integrated radiation lengths',text)
                self.assertIn('Gray: untraced directions',text)


if __name__ == '__main__':
    unittest.main()
