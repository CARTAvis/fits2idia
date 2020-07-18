#!/usr/bin/env python3

import os
import sys
import itertools
import subprocess
import warnings
import random
import re
import argparse
from collections import defaultdict
from timeit import default_timer as timer

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
    if h5diff.returncode == 0:
        return
    
    if not ignore_converter_attrs:
        assert False, fail_msg
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
            print("(Permitted converter version attributes have been ignored.)")
            assert False, fail_msg

def make_image(outfile, *dims, **params):
    cmd = ["make_image.py", "-o", outfile]
    for name, values in params.items():
        if values:
            cmd.append(name)
            try:
                for value in values:
                    cmd.append(str(value))
            except:
                cmd.append(str(values))
    cmd.append("--")
    cmd.extend(str(d) for d in dims)
    
    print(*cmd)
    
    result = subprocess.run(cmd)
    assert result.returncode == 0, "Image generation failed."
    
def convert(infile, outfile, slow=False, executable="hdf_convert"):
    cmd = [executable]
    if slow:
        cmd.append("-s")
    cmd.extend(["-o", outfile, infile])
    
    print(*cmd)
    
    result = subprocess.run(cmd)
    assert result.returncode == 0, "Conversion failed."
    
def time(infile, slow=False, executable="hdf_convert"):
    os.system("sudo sh -c 'sync && echo 3 > /proc/sys/vm/drop_caches'")
    
    cmd = [executable]
    if slow:
        cmd.append("-s")
    cmd.extend(["-o", "TIMED.hdf5", infile])
    
    print(*cmd)
    
    start = timer()
    result = subprocess.run(cmd)
    end = timer()
    
    if result.returncode > 0:
        return None

    subprocess.run(["rm", "test.fits", "TIMED.hdf5"])
        
    return end - start

def test_converter_correctness(infile):
    convert(infile, "FAST.hdf5")
    convert(infile, "SLOW.hdf5", True)
    
    # Do this first so that we fail more quickly
    compare_hdf5_hdf5("FAST.hdf5", "SLOW.hdf5", "Fast and slow versions differ.", False)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        compare_fits_hdf5("test.fits", "FAST.hdf5")
    
    subprocess.run(["rm", "test.fits", "FAST.hdf5", "SLOW.hdf5"])

def test_new_old_converter(infile, slow, old_converter):
    convert(infile, "OLD.hdf5", slow=slow, executable=old_converter)
    convert(infile, "NEW.hdf5", slow=slow)
    
    compare_hdf5_hdf5("OLD.hdf5", "NEW.hdf5", "Old and new versions differ.", True)

def small_nans_image_set():
    image_set = []
    
    for N in (2, 3, 4):
        for nans in itertools.chain((None, ("image",)), itertools.chain.from_iterable(itertools.combinations(("pixel", "row", "column", "channel", "stokes"), n) for n in range(1, 5+1))):
            for nan_density in (0, 33, 66, 100):
                dims = [random.randint(10, 50), random.randint(10, 50)]
                if N > 2:
                    dims.append(random.randint(50, 100))
                if N > 3:
                    dims.append(random.randint(1, 4))
                    
                params = {
                    "--nans": nans,
                    "--nan-density": nan_density
                }
                
                image_set.append((dims, params))
    return image_set

def large_image_set():
    image_set = []
    
    # A few bigger images to test multiple mipmaps
    for dims in ((600, 200), (600, 200, 10), (5000, 200, 10, 2)):
        for nans in (("pixel",),):
            for nan_density in (50,):
                params = {
                    "--nans": nans,
                    "--nan-density": nan_density
                }
                
                image_set.append((dims, params))
    return image_set

def timer_image_set(slow=False):
    if slow:
        return [
            ((500, 500, 1000), {}),
            ((1000, 1000, 1000), {}),
            ((5000, 5000, 500), {}),
            ((5000, 5000, 1000), {}),
            ((10000, 10000, 500), {}),
            ((10000, 10000, 1000), {}),
        ]
    else:
        return [
            ((50, 50, 50000), {}),
            ((100, 100, 50000), {}),
            ((500, 500, 500), {}),
            ((500, 500, 1000), {}),
            ((1000, 1000, 500), {}),
            ((5000, 5000, 10), {}),
        ]

# for testing this script
def dummy_image_set():
    return [
        ((10, 10, 20), {}),
        ((20, 20, 40), {}),
        ((40, 40, 80), {}),
    ]

def test_correctness(*image_sets):
    for image_set in image_sets:
        for dims, params in image_set:
            make_image("test.fits", *dims, **params)
            test_converter_correctness("test.fits")
        
def test_consistency(old_converter, *image_sets, **kwargs):
    for image_set in image_sets:
        for dims, params in image_set:
            make_image("test.fits", *dims, **params)
            test_new_old_converter("test.fits", kwargs["slow"], old_converter)
            
def test_speed(*image_sets, **kwargs):
    executables = ["hdf_convert"]
    if "compare" in kwargs:
        executables.append(kwargs["compare"])
    
    times = defaultdict(lambda: defaultdict(list))
    
    for i in range(kwargs.get("repeat", 5)):
        for executable in executables:
            for image_set in image_sets:
                for dims, _ in image_set: # we only care about the dimensions
                    make_image("test.fits", *dims)
                    t = time("test.fits", slow=kwargs["slow"], executable=executable)
                    if t is not None:
                        times[dims][executable].append(t)
    
    winners = {}
    xs = set()
    zs = set()
    
    print("Image dimensions", *executables, sep='\t')
    print()
    
    for dims in sorted(times):
        best = [np.min(times[dims][e]) for e in executables]
        print(*dims, *best, sep='\t')
        x, _, z = dims
        xs.add(x)
        zs.add(z)
        winners[(x, z)] = "A" if min(best) == best[0] else "B"
    
    if len(executables) > 1:
        xs = sorted(xs)
        zs = sorted(zs)
        
        for label, path in zip(["A", "B"], executables):
            print("%s: %s" % (label, path))
        print()
        
        print('X \ Z', *zs, sep='\t')
        
        for x in xs:
            print(x, end='\t')
            for z in zs:
                if (x, z) in winners:
                    print(winners[(x, z)], end='\t')
                else:
                    print('-', end='\t')
            print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test for the HDF5 converter")
    parser.add_argument('-c', '--compare', help='A path to another converter executable to use as a reference. If a reference converter is given, random files converted using the fast algorithm with both converters will be compared to each other. If none is given, the output of the converter at the default path will be checked for correctness (THIS IS SLOW), and its fast and slow algorithm outputs will be compared for consistency.')
    parser.add_argument('-t', '--time', action='store_true', help='Time the converter(s) instead of checking the output.')
    parser.add_argument('-s', '--slow', action='store_true', help='Use the slow converter versions when comparing converters or timing converter(s).')
    args = parser.parse_args()
    
    if args.time:
        if args.compare:
            test_speed(timer_image_set(args.slow), compare=args.compare, repeat=3, slow=args.slow)
            #test_speed(dummy_image_set(), compare=args.compare, repeat=2, slow=args.slow)
        else:
            test_speed(timer_image_set(args.slow), slow=args.slow)
            #test_speed(dummy_image_set(), slow=args.slow)
    else:
        if args.compare:
            test_consistency(args.compare, small_nans_image_set(), large_image_set(), slow=args.slow)
        else:
            test_correctness(small_nans_image_set(), large_image_set())
