#!/usr/bin/env python3

import os
import sys
import itertools
import subprocess
import warnings
import random
import re
import argparse

from astropy.io import fits
import h5py
import numpy as np
from numpy.testing import assert_equal, assert_allclose

def pprint_sparse(a):
    return "\n".join(["%r: %g" % (i, v) for i, v in np.ndenumerate(a) if v])

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
    
    width, height, depth, stokes = dims["X"], dims["Y"], dims.get("Z", 1), dims.get("W", 1)

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
        
        # CHECK BASIC STATS
        
        def assert_close(stat, func, data=fitsdata):
            if stat in sdata:
                assert_allclose(sdata[stat], func(data, axis=stats_axis), rtol=1e-5, err_msg = "%s/%s is incorrect. DIFF: %r" % (s, stat, func(data, axis=stats_axis) - sdata[stat]))
        
        assert_close("SUM", np.nansum, fitsdata.astype(np.float64))
        assert_close("SUM_SQ", np.nansum, fitsdata.astype(np.float64)**2)
        assert_close("MEAN", np.nanmean)
        assert_close("MIN", np.nanmin)
        assert_close("MAX", np.nanmax)
        
        assert (np.count_nonzero(np.isnan(fitsdata), axis=stats_axis) == sdata["NAN_COUNT"]).all(), "%s/NAN_COUNT is incorrect." % s
        
        if s in ["XY", "XYZ"]:
            # CHECK HISTOGRAMS
            
            hist = np.array(sdata["HISTOGRAM"])
            num_bins = int(max(np.sqrt(width * height), 2))
            d = np.array(hdf5data)
            
            # Note: we produce a zero histogram for datasets with only one not-nan value. Numpy changes the bounds when calculating the histogram for a single value, so it always bins it somewhere.
            if ndim == len(s):
                d = d[~np.isnan(d)]
                reference = np.histogram(d.astype(np.float64), bins=num_bins)[0] if d.size > 1 else np.zeros(num_bins)
            else:
                stats_shape = tuple(dims[a] for a in s)
                stats_merged_shape = np.multiply.reduce(stats_shape)
                rest_shape = tuple(dims[a] for a in reversed(axis_names) if a not in s)
                rest_merged_shape = np.multiply.reduce(rest_shape)
                hist_shape = rest_shape + (num_bins,)
                                            
                reference = np.apply_along_axis(lambda a: np.histogram(a[~np.isnan(a)].astype(np.float64), bins=num_bins)[0] if a[~np.isnan(a)].size > 1 else np.zeros(num_bins), 1, d.reshape(rest_merged_shape, stats_merged_shape)).reshape(hist_shape)
            
            diff = reference - hist
            
            assert hist.shape == reference.shape, "%s histogram shape %r does not match expected shape %r" % (s, hist.shape, reference.shape)
            # Note: we can't replicate the converter's binning exactly because of a precision issue, so there are occasional off-by-one bin increments.
            assert diff.sum() == 0 and diff[diff != 0].size / diff.size < 0.01, "Too many %s histogram values do not match expected values.\nSum of difference: %d\nDifference:\n%s" % (s, diff.sum(), pprint_sparse(diff))
    
    # CHECK MIPMAPS
    
    for mname, mipmap in hdf5file["0/MipMaps"]["DATA"].items():
        factor = int(re.match(r"DATA_XY_(\d+)", mname).group(1))
        mheight, mwidth = mipmap.shape[-2:]
        
        assert mwidth == np.ceil(width / factor) and mheight == np.ceil(height / factor), "Dimensions of mipmap %s are incorrect. Expected: %r Got: %r" % (mname, d.shape[:-2] + (mheight, mwidth), mipmap.shape)
    
        # TODO check mipmap contents
    # TODO check that the last mipmap is small enough
        
        
        

    fitsfile.close()
    hdf5file.close()

def test_random_image(dims, nans, nan_density):
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

def test_random_files():
    for N in (2, 3, 4):
        for nans in itertools.chain((None, ("image",)), itertools.chain.from_iterable(itertools.combinations(("pixel", "row", "column", "channel", "stokes"), n) for n in range(1, 5+1))):
            for nan_density in (0, 33, 66, 100):
                
                dims = [random.randint(10, 50), random.randint(10, 50)]
                if N > 2:
                    dims.append(random.randint(50, 100))
                if N > 3:
                    dims.append(random.randint(1, 4))
                
                test_random_image(dims, nans, nan_density)
    
    # A few bigger images to test number of mipmaps
    for dims in ((5000, 1000), (5000, 1000, 10), (5000, 1000, 10, 3)):
        for nans in (("pixel",),):
            for nan_density in (50,):
                test_random_image(dims, nans, nan_density)
                
    subprocess.run(["rm", "test.fits", "FAST.hdf5", "SLOW.hdf5"])
    
def test_specific_files(args):
    fitsfile = args.fitsfile or args.hdf5file.replace("fits", "hdf5")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        compare_fits_hdf5(fitsfile, args.hdf5file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test for the HDF5 converter")
    parser.add_argument('-d', '--hdf5file', help='Converted HDF5 filename. If no file is given, a set of randomly generated files with different dimensions and different patterns of NaN values will be tested.')
    parser.add_argument('-f', '--fitsfile', help="Original FITS filename. The default is the same name as the HDF5 file with the suffix replaced.")
    args = parser.parse_args()
    
    if args.hdf5file:
        test_specific_files(args)
    else:
        test_random_files()
