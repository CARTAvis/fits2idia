#!/usr/bin/env python3

import os
import sys
import itertools
import subprocess
import warnings
import random

from astropy.io import fits
import h5py
import numpy as np
from numpy.testing import assert_equal, assert_allclose

def compare_fits_hdf5(fitsname, hdf5name):

    fitsfile = fits.open(fitsname)
    hdf5file = h5py.File(hdf5name)

    fitsdata = fitsfile[0].data
    hdf5data = hdf5file["0/DATA"]

    # CHECK MAIN DATASET

    assert_equal(hdf5data, fitsdata, err_msg="Main dataset differs.")

    ndim = fitsdata.ndim

    if ndim == 2:
        axis_names = "XY"
    elif ndim == 3:
        axis_names = "XYZ"
    elif ndim == 4:
        axis_names = "XYZW"
    else:
        sys.exit("Error: unsupported image dimensions.")

    axes = {v: k for k, v in enumerate(reversed(axis_names))}
    dims = {v: fitsdata.shape[k] for k, v in enumerate(reversed(axis_names))}

    # CHECK SWIZZLES

    swizzled_name = None

    if ndim == 3:
        swizzled_name = "ZYX"
    elif ndim == 4:
        swizzled_name = "ZYXW"

    if swizzled_name:
        swizzled_shape = tuple(dims[a] for a in reversed(swizzled_name))    
        assert swizzled_name in hdf5file["0/SwizzledData"], "No swizzled dataset found."
        assert hdf5file["0/SwizzledData"][swizzled_name].shape == swizzled_shape, "Swizzled dataset has incorrect dimensions."

    # CHECK STATS

    stats = ["XY"]

    if ndim >= 3 and dims["Z"] > 1:
        stats.append("Z")
        stats.append("XYZ")
        
    for s in stats:
        assert s in hdf5file["0/Statistics"], "%s statistics missing." % s
        
        sdata = hdf5file["0/Statistics"][s]
        stats_axis = tuple(axes[a] for a in s)
        
        def assert_close(stat, func, data=fitsdata):
            if stat in sdata:
                assert_allclose(sdata[stat], func(data, axis=stats_axis), rtol=1e-5, err_msg = "%s/%s is incorrect. DIFF: %r" % (s, stat, func(data, axis=stats_axis) - sdata[stat]))
        
        assert_close("SUM", np.nansum, fitsdata.astype(np.float64))
        assert_close("SUM_SQ", np.nansum, fitsdata.astype(np.float64)**2)
        assert_close("MEAN", np.nanmean)
        assert_close("MIN", np.nanmin)
        assert_close("MAX", np.nanmax)
        
        assert (np.count_nonzero(np.isnan(fitsdata), axis=stats_axis) == sdata["NAN_COUNT"]).all(), "%s/NAN_COUNT is incorrect." % s
        
    # CHECK HISTOGRAMS

    # TODO: check for bin size correctness
    # TODO: check histogram values
    # TODO: check mipmaps

    fitsfile.close()
    hdf5file.close()

if __name__ == "__main__":
    for N in (2, 3, 4):
        for nans in itertools.chain((None, ("image",)), itertools.chain.from_iterable(itertools.combinations(("row", "column", "channel", "stokes"), n) for n in range(1, 4+1))):
            for nan_density in (0, 33, 66, 100):
                
                dims = [random.randint(10, 50), random.randint(10, 50)]
                if N > 2:
                    dims.append(random.randint(50, 100))
                if N > 3:
                    dims.append(random.randint(1, 4))
                
                print("Testing %r image" % (dims,), end="")
                
                params = ["make_image.py", "-o", "test.fits"]
                if nans:
                    params.extend(("--nans", *(str(n) for n in nans), "--nan-density", str(nan_density), "--"))
                    print(" with NaNs inserted in random %r with density %d" % (nans, nan_density), end="")
                params.extend(str(d) for d in dims)
                
                print("...")
                                
                subprocess.run(params)
                
                subprocess.run(["hdf_convert", "-q", "-o", "FAST.hdf5", "test.fits"])
                subprocess.run(["hdf_convert", "-q", "-s", "-o", "SLOW.hdf5", "test.fits"])
                
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    compare_fits_hdf5("test.fits", "FAST.hdf5")
                
                h5diff = subprocess.run(["h5diff", "FAST.hdf5", "SLOW.hdf5"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                assert h5diff.returncode == 0, "Fast and slow versions differ."
                
    subprocess.run(["rm", "test.fits", "FAST.hdf5", "SLOW.hdf5"])
                    
    
