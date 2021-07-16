import matplotlib
matplotlib.use('agg') # use matplotlib in command line mode
import warnings
import numpy as np
from matplotlib.pyplot import imsave
import nibabel as nib
from nibabel import processing as nibproc # used for resample images

# hard coded character '0'~'9'.
glyph = [
    np.transpose(np.array([[0,0,0,0,0,0],[0,0,1,1,0,0],[0,1,0,0,1,0],[0,1,0,0,1,0],[0,1,0,0,1,0],[0,1,0,0,1,0],[0,0,1,1,0,0],[0,0,0,0,0,0]])),
    np.transpose(np.array([[0,0,0,0,0,0],[0,0,1,1,0,0],[0,1,0,1,0,0],[0,0,0,1,0,0],[0,0,0,1,0,0],[0,0,0,1,0,0],[0,1,1,1,1,0],[0,0,0,0,0,0]])),
    np.transpose(np.array([[0,0,0,0,0,0],[0,0,1,1,0,0],[0,1,0,0,1,0],[0,0,0,0,1,0],[0,0,0,1,0,0],[0,0,1,0,0,0],[0,1,1,1,1,0],[0,0,0,0,0,0]])),
    np.transpose(np.array([[0,0,0,0,0,0],[0,1,1,1,0,0],[0,0,0,0,1,0],[0,0,1,1,0,0],[0,0,0,0,1,0],[0,0,0,0,1,0],[0,1,1,1,0,0],[0,0,0,0,0,0]])),
    np.transpose(np.array([[0,0,0,0,0,0],[0,0,0,1,0,0],[0,0,1,1,0,0],[0,1,0,1,0,0],[1,1,1,1,1,0],[0,0,0,1,0,0],[0,0,0,1,0,0],[0,0,0,0,0,0]])),
    np.transpose(np.array([[0,0,0,0,0,0],[0,1,1,1,1,0],[0,1,0,0,0,0],[0,1,1,1,0,0],[0,0,0,0,1,0],[0,0,0,0,1,0],[0,1,1,1,0,0],[0,0,0,0,0,0]])),
    np.transpose(np.array([[0,0,0,0,0,0],[0,0,1,1,1,0],[0,1,0,0,0,0],[0,1,1,1,0,0],[0,1,0,0,1,0],[0,1,0,0,1,0],[0,0,1,1,0,0],[0,0,0,0,0,0]])),
    np.transpose(np.array([[0,0,0,0,0,0],[0,1,1,1,1,0],[0,1,0,0,1,0],[0,0,0,0,1,0],[0,0,0,1,0,0],[0,0,1,0,0,0],[0,1,0,0,0,0],[0,0,0,0,0,0]])),
    np.transpose(np.array([[0,0,0,0,0,0],[0,0,1,1,0,0],[0,1,0,0,1,0],[0,0,1,1,0,0],[0,1,0,0,1,0],[0,1,0,0,1,0],[0,0,1,1,0,0],[0,0,0,0,0,0]])),
    np.transpose(np.array([[0,0,0,0,0,0],[0,0,1,1,0,0],[0,1,0,0,1,0],[0,1,0,0,1,0],[0,0,1,1,1,0],[0,0,0,0,1,0],[0,1,1,1,0,0],[0,0,0,0,0,0]]))
]

# _rect_intersect():
# utility function to calculate the intersection of the two rectangles (x,y,w,h)
# original source is from:
# https://answers.opencv.org/question/90455/how-to-perform-intersection-or-union-operations-on-a-rect-in-python/
def _rect_intersect(a, b):
    '''
    a: (x1,y1,w1,h1)
    b: (x2,y2,w2,h2)
    '''
    x = max(a[0], b[0])
    y = max(a[1], b[1])
    w = min(a[0]+a[2], b[0]+b[2]) - x
    h = min(a[1]+a[3], b[1]+b[3]) - y
    if w<0 or h<0: return (x,y,0,0) # or (0,0,0,0) ?
    return (x, y, w, h)

# _paste_slice(): copy and paste an image slice to another image
# here (x,y) indicates the upper-left coordinates of the destination position 
# this function is designed to be robust and it can handle out of bound values
# carefully.
def _paste_slice(src, dst, x, y):
    '''
    slice shape: (MxN) or (MxNx3) for a RGB slice
    image shape: (MxNx3)
    '''
    shp = src.shape
    if x>=0 and y>=0 and x+shp[0]<=dst.shape[0] and y+shp[1]<=dst.shape[1]:
        if len(src.shape)==2:
            for ch in range(3): # channel broadcast
                dst[x:x+shp[0],y:y+shp[1],ch]=src
        else:
            dst[x:x+shp[0],y:y+shp[1]]=src
        return dst
    else:
        # out of bounds
        src_rect = (x,y,shp[0],shp[1])
        dst_rect = (0,0,dst.shape[0],dst.shape[1])
        ist_rect = _rect_intersect(src_rect, dst_rect)
        if ist_rect[2] == 0 or ist_rect[3] == 0:
            return dst # no intersection
        src_offset = ( ist_rect[0]-x, ist_rect[1]-y )
        if len(src.shape)==2:
            for ch in range(3): # channel broadcast
                dst[ist_rect[0]:ist_rect[0]+ist_rect[2], ist_rect[1]:ist_rect[1]+ist_rect[3], ch ]=\
                    src[src_offset[0]:src_offset[0]+ist_rect[2], src_offset[1]:src_offset[1]+ist_rect[3]]
        else:
            dst[ist_rect[0]:ist_rect[0]+ist_rect[2], ist_rect[1]:ist_rect[1]+ist_rect[3]]=\
                src[src_offset[0]:src_offset[0]+ist_rect[2], src_offset[1]:src_offset[1]+ist_rect[3]]
        return dst

def _load_nifti(nii_path):
    return nib.load(nii_path).get_fdata().astype('float32')

# original answer from:
# https://stackoverflow.com/questions/45027220/expanding-zooming-in-a-numpy-array
# ratio must be an integer
def _int_zoom(array, ratio):
    return np.kron(array, np.ones((ratio,ratio)))

def lightbox(
        # 1. basic options:
        nii_file, n_rows, n_cols, save_path, slice_range=None, slice_step=None, view_axis='axial', 
        # 2. color overlay options:
        # use these options if you want to draw color overlays (ie, draw lesion overlay)
        # you can assign each value in mask to have a different color (except black)
        nii_mask=None, color_palette=None, blend_weight=0.5,
        # 3. resample options:
        # use this option if your image resolution is not isotropic, ie: resample=1.0 to 
        # resample image to have 1mm isotropic resolution. resample_order can be a integer
        # from 0~5. resample_order=0 means nearest interpolation, 1 means linear interpolation.
        # maximum order is up to 5.
        resample=None, resample_order=1,
        # 4. miscellaneous options below
        show_slice_number=True, font_size=1):
    
    nii_data = _load_nifti(nii_file)

    assert view_axis in ['sagittal', 'coronal', 'axial'], '"view_axis" should be "sagittal", "coronal" or "axial"(default).'
    assert len(nii_data.shape) == 3, 'Only support 3D NIFTI data.'
    assert slice_range is None or isinstance(slice_range, (tuple, list)), 'Invalid "slice_range" setting.'
    assert len(save_path)>4 and save_path[-4:]=='.png', '"save_path" must ends with ".png".'
    assert isinstance(font_size,int), '"font_size" must be "int".'

    if nii_mask is not None:
        nii_mask_data = np.around(_load_nifti(nii_mask)).astype('int')
        assert nii_mask_data.shape == nii_data.shape, 'data shape is not equal to mask shape.'
        assert color_palette is not None, 'must assign color palette when mask is provided.'
        assert nii_mask_data.max() == len(color_palette), 'Invalid color palette.'
    
    # nii_data: intensity image
    # nii_mask_data: color overlay, color is assigned by user defined color palette


    # resample image if needed.
    # this happens when you want to show a NIFTI image with anisotropic resolution
    # if you don't resample the image to a isotropic resolution then the image will 
    # appears to be stretched
    if resample is not None:
        if slice_range is not None or slice_step is not None:
            warnings.warn('Image will be resampled. "slice_range" and "slice_step" settings '
                          'are no longer accurate.')
        # reload nii data from resampled image
        resampled_data = nibproc.resample_to_output(nib.load(nii_file), [resample, resample, resample], order=resample_order) # linear interpolation
        nii_data = resampled_data.get_fdata().astype('float32') 
        # reload nii mask from resampled image
        resampled_mask = nibproc.resample_to_output(nib.load(nii_mask), [resample, resample, resample], order=0) # force using nearest interpolation
        nii_mask_data = np.around(resampled_mask.get_fdata()).astype('int') # to avoid floating point rounding error, here we used np.around()

    # OK, now all preparation works have been done

    if view_axis == 'sagittal':
        nii_slices = nii_data.shape[0]
        nii_data_tr = np.transpose(nii_data, [0,1,2])[::-1,::-1,::-1]
        if nii_mask is not None:
            nii_mask_data_tr = np.transpose(nii_mask_data, [0,1,2])[::-1,::-1,::-1]
    elif view_axis == 'coronal':
        nii_slices = nii_data.shape[1]
        nii_data_tr = np.transpose(nii_data, [1,0,2])[::-1,::-1,::-1]
        if nii_mask is not None:
            nii_mask_data_tr = np.transpose(nii_mask_data, [1,0,2])[::-1,::-1,::-1]
    elif view_axis == 'axial':
        nii_slices = nii_data.shape[2]
        nii_data_tr = np.transpose(nii_data, [2,0,1])
        if nii_mask is not None:
            nii_mask_data_tr = np.transpose(nii_mask_data, [2,0,1])

    # calculate slice start and slice end
    if slice_range is None:
        slice_start, slice_end = 0, nii_slices-1
    else:
        slice_start, slice_end = slice_range[0], slice_range[1]
    if slice_end is None or slice_end<0 or slice_end>=nii_slices:
        slice_end = nii_slices-1
    if slice_start is None or slice_start<0 or slice_start>=nii_slices:
        slice_end = 0
    if slice_start > slice_end:
        slice_start = slice_end
    total_slices = slice_end - slice_start + 1
    view_slices = n_rows * n_cols
    
    # calculate slice step
    if view_slices >= total_slices:
        slice_step = 1
    else:
        if slice_step is None:
            slice_step = float(total_slices) / float(view_slices)
    
    # calculate slice shape and image size
    slice_shape = (nii_data_tr.shape[1], nii_data_tr.shape[2])
    image_height = n_rows * slice_shape[1]
    image_width = n_cols * slice_shape[0]

    image = np.zeros([image_width, image_height, 3]) # RGB channel

    # normalize nii intensity
    nii_data_tr = (nii_data_tr - nii_data_tr.min()) / (nii_data_tr.max()-nii_data_tr.min() + 0.00001)

    current_slice = slice_start
    for iy in range(n_rows):
        for ix in range(n_cols):
            if current_slice < nii_data_tr.shape[0]:
                # paste slice data
                slice_data = nii_data_tr[current_slice]
                if nii_mask is not None:
                    ind_data = nii_mask_data_tr[current_slice]
                    color_data = np.zeros([slice_data.shape[0], slice_data.shape[1], 3])
                    for ic in range(len(color_palette)):
                        color_data[:,:,0] = np.where(ind_data==ic+1, slice_data * (1-blend_weight) + color_palette[ic][0]/255.0 * blend_weight , slice_data) # fill R
                        color_data[:,:,1] = np.where(ind_data==ic+1, slice_data * (1-blend_weight) + color_palette[ic][1]/255.0 * blend_weight , slice_data) # fill G
                        color_data[:,:,2] = np.where(ind_data==ic+1, slice_data * (1-blend_weight) + color_palette[ic][2]/255.0 * blend_weight , slice_data) # fill B 
                    _paste_slice(color_data, image, slice_shape[0]*ix, slice_shape[1]*iy)
                else:
                    _paste_slice(nii_data_tr[current_slice], image, slice_shape[0]*ix, slice_shape[1]*iy)
                if show_slice_number:
                    # paste slice number
                    slice_str = '%03d' % current_slice
                    numbering_pos = (slice_shape[0]*ix+2, slice_shape[1]*iy+2)
                    for ig in range(3): # max 3 digits
                        selected_glyph = glyph[int(slice_str[ig])]
                        selected_glyph = _int_zoom(selected_glyph, font_size)
                        _paste_slice(selected_glyph, image, numbering_pos[0]+ig*6*font_size, numbering_pos[1])
            current_slice = int(current_slice + slice_step)

    imsave(save_path, np.transpose(image,[1,0,2])) # dont forget to transpose the matrix before saving!
    