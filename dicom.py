#!/usr/bin/env python
# coding: utf-8

"""
A module for reading and writing DICOM files, basically a wrapper for calls on
the GDCM (Grassroots DICOM) library [1, 2].

This module borrows from the GDCM "ConvertNumpy" example [3] in its 
<__to_dtype()> and <__to_array()> functions.

References
[1] http://gdcm.sourceforge.net/ (20120314).
[2] http://gdcm.sourceforge.net/html/ (20120314)
[3] http://gdcm.sourceforge.net/html/ConvertNumpy_8py-example.html (20120314)
"""

from __future__ import division

import gdcm
import numpy as np
import os
import string


def open_image(path, verbose=True):
    """
    Open a two- or three-dimensional DICOM image at the given <path>.
    
    Return a tuple (ireader, data) where <ireader> represents a
    <gdcm.ImageReader> reference to the image as returned by the GDCM library,
    and <data> holds the actual image data in a Numpy array.
    
    Note that the image data as a <gdcm.Image> instance can be obtained from
    the <ireader> via <ireader.GetImage()>. However, the image instance is
    only valid as long as the <ireader> is not destroyed. This is the reason
    why the <ImageReader> rather than the <Image> instance is returned here.
    
    If unsuccessful, an <IOError> will be raised.
    """
    # Create a reader for the image
    reader = gdcm.ImageReader()
    reader.SetFileName(path)
    # Try to actually read it
    read_success = reader.Read()
    if not read_success:
        raise IOError("Error opening dicom-file.")
    image = reader.GetImage()
    # Convert to Numpy array
    try:
        data = __to_array(image)
    except ValueError:
        # Re-raise as IOError with complete traceback
        import sys
        (value, traceback) = sys.exc_info()[1:]
        raise (IOError, value, traceback)
    if verbose:
        print "Image loaded:", path
        print "Meta data:"
        print image
    return (reader, data)


def open_stack(path, prefix="", postfix=""):
    """
    Open a list of two-dimensional DICOM files at the directory specified via
    <path> and build a three-dimensional image stack from them.
    
    Read the files from the specified directory in alphabetical order (case-
    insensitive) and stack their image data along the "zeroth" axis (axis count
    starts with 0) of a three-dimensional Numpy array. Only images with one
    sample per pixel (i.e. grayscale) images are supported. If a <prefix> is
    given, only files are taken into account whose name starts with the
    <prefix> (case-sensitive). The same applies to file name endings and a
    given <postfix>. A combination is also possible, in this case both <prefix>
    and <postfix> have to match.
    
    Return a tuple (ireaders, data) where <ireaders> represents a list of
    <gdcm.ImageReader> references corresponding to the image data in the
    respective slice of <data>. The <data> variable, in turn, holds the actual
    image data in a Numpy array.
    
    If unsuccessful, an <IOError> will be raised.
    
    See also: <open_image()>.
    """
    ireaders = []
    data = []
    
    # Browse the contents of the given directory in alphabetical order (case-
    # insensitive or otherwise all files with names starting with A-Z would
    # appear before file names starting with a-z etc.)
    for entry in sorted(os.listdir(path), key=str.lower):
        
        # Get absolute path to entry
        abs_entry = os.path.join(path, entry)
        
        if os.path.isdir(abs_entry):
            # Check if the current entry is a file
            continue
            
        if not entry.startswith(prefix) or not entry.endswith(postfix):
            # Check file name
            continue
        
        # Get the actual image data
        (ireader, islice) = open_image(abs_entry, verbose=False)

        if islice.ndim != 2 or ireader.GetImage().GetNumberOfDimensions() != 2:
            # Check if the image is two-dimensional (check the dimensionality
            # of the array; also check the original image's number of
            # dimensions, as a two-dimensional array could be the result of a
            # one-dimensional image with more than one sample per pixel)
            raise IOError("Error: Image is not 2-D.")
        
        ireaders.append(ireader)
        data.append(islice)
    
    return (ireaders, np.asarray(data))


def pre_and_postfix_from_file(path):
    """
    Get the prefix and the postfix for the given <path>.
    
    By "postfix", we refer to the letters after the rightmost full stop (<.>)
    in the file name. By "prefix", we refer to the string that remains when
    cropping the postfix and the rightmost full stop (if present), and then
    successively cropping digits from the right as far as possible without
    hitting a non-digit character.
    
    Absolute paths will be cropped before processing.
    
    Return a tuple (prefix, postfix).
    
    For example, "/home/username/series0_0001.img" will be returned as
    ("series0_", "img"); "0001.nii" will be returned as ("", "nii");
    "image0001" will be returned as ("image", "").
    """
    # If an absolute path is given, crop it to the actual file name first
    path = path.rsplit(os.sep, 1)[-1]
    
    # Get the prefix (crop trailing digits) and the postfix (if present)
    split_name = path.rsplit(".", 1)
    prefix = split_name[0].rstrip(string.digits)
    postfix = split_name[1] if len(split_name) == 2 else ""
    
    return (prefix, postfix)
    

def __to_dtype(pixel_format):
    """
    Return the matching numpy data type (<dtype> instance) for the given
    <pixel_format> (<gdcm.PixelFormat> instance expected).
    
    Raise a <ValueError> if the <pixel_format> could not be converted.
    """
    scalar_type = pixel_format.GetScalarType()
    scalar_string = pixel_format.GetScalarTypeAsString()

    # Encode the actual mapping
    # Not supported: UINT12, INT12, FLOAT16 (no Numpy equivalent present)
    switch = {gdcm.PixelFormat.UINT8: np.uint8,
              gdcm.PixelFormat.UINT16: np.uint16,
              gdcm.PixelFormat.UINT32: np.uint32,
              gdcm.PixelFormat.INT8: np.int8,
              gdcm.PixelFormat.INT16: np.int16,
              gdcm.PixelFormat.INT32: np.int32,
              gdcm.PixelFormat.FLOAT32: np.float32,
              gdcm.PixelFormat.FLOAT64: np.float64}
    try:
        dtype = switch[scalar_type]
    except KeyError:
        raise ValueError("Error: Unsupported file type %s." % scalar_string)

    return np.dtype(dtype)


def __to_array(image):
    """
    Convert the given <image> (GDCM image expected) to a Numpy array.
    
    Return a Numpy array that keeps the original <image>'s data type (or
    rather, holds the respective Numpy data type).
    
    Raise a <ValueError> if the data type of the <image> or its number of
    dimensions is not supported. Two or three dimensions are supported with
    a number of values per pixel/voxel >= 1. If the number of values per
    pixel/voxel is > 1, an extra dimension is added to the result array.
    """
    # Get the data type of the image and try to convert to Numpy data type
    pixel_format = image.GetPixelFormat()
    dtype = __to_dtype(pixel_format)

    if image.GetNeedByteSwap():
        # Mark byte swap if necessary (I hope GetNeedByteSwap does what its
        # name implies -- unfortunately no documentation is present). Note that
        # this code has not been tested
        dtype = dtype.newbyteorder('S')
    
    # Get the necessary shape information from the image and convert it as
    # needed by Numpy
    dimensions = image.GetNumberOfDimensions()
    spp = pixel_format.GetSamplesPerPixel()
    if dimensions == 2 or dimensions == 3:
        shape = []
        for i in range(dimensions):
            shape.append(image.GetDimension(i))
        if spp != 1:
            shape.append(spp)
        # Not sure why the order of dimensions has to be inverted, but
        # definitely necessary
        shape = tuple(shape[::-1])
    else:
        raise ValueError("Error: Wrong image dimensions %d." % dimensions)
    
    # Get the actual image data, create an array from it, reshape and return
    img_buffer = image.GetBuffer()
    result = np.frombuffer(img_buffer, dtype=dtype)
    result.shape = shape
    
    return result


if __name__ == "__main__":
    """
    Some "testing" code
    """
    import matplotlib.pyplot as plt
    
    filename = "/home/simon/Data/20120314-AllScanRescanTestSubjects/CMS01_20120110/CMS01_20120110_0004_MPRAGE_1x1x1mm3_1/CMS01_0004_000001.ima"
    data = open_image(filename)[1]

    plt.imshow(data)
    plt.colorbar()

    plt.set_cmap("gray")
    plt.show()

    dirname = "/home/simon/Data/20120314-AllScanRescanTestSubjects/CMS01_20120110/CMS01_20120110_0004_MPRAGE_1x1x1mm3_1/"
    (readers, data) = open_stack(dirname, postfix=".ima")
    print data
    pass
