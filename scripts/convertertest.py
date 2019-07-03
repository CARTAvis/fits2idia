#!/usr/bin/env python3

import os
import sys
import argparse
from astropy.io import fits
import h5py
import numpy as np

parser = argparse.ArgumentParser(description="Test for the HDF5 converter")
parser.add_argument('filename', help='Converted HDF5 filename')
parser.add_argument('-o', '--original', help="Original FITS filename. The default is the same name as the HDF5 file with the suffix replaced.")
args = parser.parse_args()

hdf5name = args.filename
fitsname = args.original

if not fitsname:
    basefilename, _ = os.path.splitext(hdf5name)
    fitsname = basefilename + ".fits"

fitsfile = fits.open(fitsname)
hdf5file = h5py.File(hdf5name)

fitsdata = fitsfile[0].data
hdf5data = hdf5file["0/DATA"]

# CHECK MAIN DATASET

assert fitsdata.shape == hdf5data.shape, "Main dataset has incorrect dimensions."
assert (fitsdata == hdf5data).all(), "Main dataset differs."

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

if ndim == 3:
    swizzled_name = "ZYX"
elif ndim == 4:
    swizzled_name = "ZYXW"
    
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
    
    if "SUM" in sdata:
        assert np.abs(sdata["SUM"] - np.nansum(fitsdata, axis=stats_axis)).max() < 1e-6, "%s/SUM is incorrect." % s
    
    if "SUM_SQ" in sdata:
        assert np.abs(sdata["SUM_SQ"] - np.nansum(fitsdata**2, axis=stats_axis)).max() < 1e-6, "%s/SUM_SQ is incorrect." % s
        
    if "MEAN" in sdata:
        assert np.abs(sdata["MEAN"] - np.nanmean(fitsdata, axis=stats_axis)).max() < 1e-6, "%s/MEAN is incorrect." % s
        
    assert np.abs(sdata["MIN"] - np.nanmin(fitsdata, axis=stats_axis)).max() < 1e-6, "%s/MIN is incorrect." % s
    assert np.abs(sdata["MAX"] - np.nanmax(fitsdata, axis=stats_axis)).max() < 1e-6, "%s/MAX is incorrect." % s
    
    assert (np.count_nonzero(np.isnan(fitsdata), axis=stats_axis) == sdata["NAN_COUNT"]).all(), "%s/NAN_COUNT is incorrect." % s
    
# CHECK HISTOGRAMS

# TODO: check for bin size correctness
# TODO: check histogram values

fitsfile.close()
hdf5file.close()
