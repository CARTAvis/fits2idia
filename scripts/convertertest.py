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
from numpy.testing import assert_equal, assert_allclose, assert_almost_equal

def pprint_sparse_diff(one, two, tolerance=1e-6):
    return "\n".join(["%r: %g" % (i, v) for i, v in np.ndenumerate(np.abs(one - two)) if v > tolerance])

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
            assert diff.sum() == 0 and diff[diff != 0].size / diff.size < 0.01, "Too many %s histogram values do not match expected values.\nSum of difference: %d\nDifference:\n%s" % (s, diff.sum(), pprint_sparse_diff(hist, reference))
    
    # CHECK MIPMAPS
    
    if "MipMaps" in hdf5file["0"]:
        for mname, mipmap in sorted(hdf5file["0/MipMaps"]["DATA"].items(), key=lambda x: x[1].size, reverse=True):
            factor = int(re.match(r"DATA_XY_(\d+)", mname).group(1))
            mheight, mwidth = mipmap.shape[-2:]
            
            assert mwidth == np.ceil(width / factor) and mheight == np.ceil(height / factor), "Dimensions of mipmap %s are incorrect. Expected: %r Got: %r" % (mname, d.shape[:-2] + (mheight, mwidth), mipmap.shape)
        
            # check mipmap contents
            
            def assert_mipmap_channel_equal(name, channel, d, got):
                got = np.array(got)
                expected = np.array([[np.nanmean(d[y*factor:(y+1)*factor, x*factor:(x+1)*factor]) for x in range(mwidth)] for y in range(mheight)])
                assert_allclose(expected, got, rtol=1e-5, atol=1e-7, equal_nan=True, err_msg = "Mipmap %s channel %r is incorrect. \nEXPECTED:\n%r\nGOT:\n%r\nDIFF:\n%s" % (name, channel, expected, got, pprint_sparse_diff(expected, got)))
            
            d = np.array(hdf5data)
            rest = mipmap.shape[:-2]
            
            if rest:
                for i in np.ndindex(rest):
                    assert_mipmap_channel_equal(mname, i, d[i], mipmap[i])
            else:
                assert_mipmap_channel_equal(mname, 0, d, mipmap)

        # check that the last mipmap is small enough
        assert mheight <= 128 and mwidth <= 128, "Smallest mipmap (%s) does not fit in 128x128 tile (dims: (%d, %d))" % (mname, mwidth, mheight)
    
    fitsfile.close()
    hdf5file.close()

def compare_hdf5_hdf5(file1, file2, fail_msg, ignore_converter_attrs=True):
    h5diff = subprocess.run(["h5diff", file1, file2], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if not ignore_converter_attrs:
        assert h5diff.returncode == 0, fail_msg
    else:
        permitted_differences = [
            "attribute: <SCHEMA_VERSION of </0>> and <SCHEMA_VERSION of </0>>",
            "attribute: <HDF5_CONVERTER of </0>> and <HDF5_CONVERTER of </0>>",
            "attribute: <HDF5_CONVERTER_VERSION of </0>> and <HDF5_CONVERTER_VERSION of </0>>",
        ]
        output = h5diff.stdout.decode('ascii')
        lines = set(output.strip().split("\n"))
        removed = 0
        for d in permitted_differences:
            if d in lines:
                lines.remove(d)
                removed += 1
        if "%d differences found" % removed not in lines:
            print(lines)
            assert false
    
def convert_fast_slow(fitsfile):
    fast = subprocess.run(["hdf_convert", "-q", "-o", "FAST.hdf5", fitsfile])
    assert fast.returncode == 0, "Fast conversion failed."
    
    slow = subprocess.run(["hdf_convert", "-q", "-s", "-o", "SLOW.hdf5", fitsfile])
    assert slow.returncode == 0, "Slow conversion failed."

def test_conversion(fitsfile, compare_to_fits=True):
    convert_fast_slow(fitsfile)
    
    if compare_to_fits:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            compare_fits_hdf5(fitsfile, "FAST.hdf5")
    
    compare_hdf5_hdf5("FAST.hdf5", "SLOW.hdf5", "Fast and slow versions differ.", False)

def test_random_image(dims, nans, nan_density):
    print("Testing %r image" % (dims,), end="")
                
    params = ["make_image.py", "-o", "test.fits"]
    if nans:
        params.extend(("--nans", *(str(n) for n in nans), "--nan-density", str(nan_density), "--"))
        print(" with NaNs inserted in random %r with density %d" % (nans, nan_density), end="")
    params.extend(str(d) for d in dims)
    
    print("...")
                    
    subprocess.run(params)
    
    test_conversion("test.fits")
    
def test_specific_image(fitsfile, hdf5file=None):
    print("Converting", fitsfile)
    test_conversion(fitsfile, hdf5file is None)
    
    if hdf5file:
        print("Comparing to reference", hdf5file)
        compare_hdf5_hdf5("FAST.hdf5", hdf5file, "Converted version differs from reference %r" % hdf5file, True)
        
    subprocess.run(["rm", "FAST.hdf5", "SLOW.hdf5"])

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
    
    # A few bigger images to test multiple mipmaps
    for dims in ((5000, 200), (5000, 200, 10), (5000, 200, 10, 2)):
        for nans in (("pixel",),):
            for nan_density in (50,):
                test_random_image(dims, nans, nan_density)
                
    subprocess.run(["rm", "test.fits", "FAST.hdf5", "SLOW.hdf5"])
    
def expand_list(globs):
    if globs is None:
        return []
    expanded = []
    for g in globs:
        expanded.extend(glob.glob(g))
    return expanded

def split_path(path):
    directory, filename = os.path.split(path)
    basefilename, extension = os.path.splitext(filename)
    return (directory, basefilename, extension)
    
def join_path(directory, basefilename, extension):
    return "/".join(directory, ".".join(basefilename, extension))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test for the HDF5 converter")
    parser.add_argument('--fits', help='A path to an input FITS file. If no HDF5 file is given, the FITS file will be converted and compared to the converted HDF5 file (THIS IS SLOW!). If an HDF5 file is given, the converted HDF5 file will be compared to the reference HDF5 file.', )
    parser.add_argument('--hdf5', help='A path to an HDF5 reference file. Ignored if no FITS files is given.')
    args = parser.parse_args()
    
    if "fits" in args:
        test_specific_image(args.fits, args.hdf5) 
    else:
        test_random_files()
