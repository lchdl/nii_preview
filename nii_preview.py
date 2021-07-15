import matplotlib
matplotlib.use('agg') 
import numpy as np
from matplotlib.pyplot import imsave
import nibabel as nib

glyph = [
    np.transpose(np.array([[0,0,0,0,0,0],[0,0,1,1,0,0],[0,1,0,0,1,0],[0,1,0,0,1,0],[0,1,0,0,1,0],[0,1,0,0,1,0],[0,0,1,1,0,0],[0,0,0,0,0,0]])),
    np.transpose(np.array([[0,0,0,0,0,0],[0,0,1,1,0,0],[0,1,0,1,0,0],[0,0,0,1,0,0],[0,0,0,1,0,0],[0,0,0,1,0,0],[0,1,1,1,1,0],[0,0,0,0,0,0]])),
    np.transpose(np.array([[0,0,0,0,0,0],[0,0,1,1,0,0],[0,1,0,0,1,0],[0,0,0,0,1,0],[0,0,0,1,0,0],[0,0,1,0,0,0],[0,1,1,1,1,0],[0,0,0,0,0,0]])),
    np.transpose(np.array([[0,0,0,0,0,0],[0,1,1,1,0,0],[0,0,0,0,1,0],[0,0,1,1,0,0],[0,0,0,1,0,0],[0,0,0,0,1,0],[0,1,1,1,0,0],[0,0,0,0,0,0]])),
    np.transpose(np.array([[0,0,0,0,0,0],[0,0,0,1,0,0],[0,0,1,1,0,0],[0,1,0,1,0,0],[1,1,1,1,1,0],[0,0,0,1,0,0],[0,0,0,1,0,0],[0,0,0,0,0,0]])),
    np.transpose(np.array([[0,0,0,0,0,0],[0,1,1,1,1,0],[0,1,0,0,0,0],[0,1,1,1,0,0],[0,0,0,0,1,0],[0,0,0,0,1,0],[0,1,1,1,0,0],[0,0,0,0,0,0]])),
    np.transpose(np.array([[0,0,0,0,0,0],[0,0,1,1,1,0],[0,1,0,0,0,0],[0,1,1,1,0,0],[0,1,0,0,1,0],[0,1,0,0,1,0],[0,0,1,1,0,0],[0,0,0,0,0,0]])),
    np.transpose(np.array([[0,0,0,0,0,0],[0,1,1,1,1,0],[0,1,0,0,1,0],[0,0,0,0,1,0],[0,0,0,1,0,0],[0,0,1,0,0,0],[0,1,0,0,0,0],[0,0,0,0,0,0]])),
    np.transpose(np.array([[0,0,0,0,0,0],[0,0,1,1,0,0],[0,1,0,0,1,0],[0,0,1,1,0,0],[0,1,0,0,1,0],[0,1,0,0,1,0],[0,0,1,1,0,0],[0,0,0,0,0,0]])),
    np.transpose(np.array([[0,0,0,0,0,0],[0,0,1,1,0,0],[0,1,0,0,1,0],[0,1,0,0,1,0],[0,0,1,1,1,0],[0,0,0,0,1,0],[0,1,1,1,0,0],[0,0,0,0,0,0]]))
]

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

def _paste_slice(slice_data, image, x, y):
    '''
    slice shape: (MxN)
    image shape: (MxNx3)
    '''
    shp = slice_data.shape
    if x>=0 and y>=0 and x+shp[0]<=image.shape[0] and y+shp[1]<=image.shape[1]:
        for ch in range(3):
            image[x:x+shp[0],y:y+shp[1],ch]=slice_data
        return image
    else:
        # out of bounds
        src_rect = (x,y,shp[0],shp[1])
        dst_rect = (0,0,image.shape[0],image.shape[1])
        ist_rect = _rect_intersect(src_rect, dst_rect)
        if ist_rect[2] == 0 or ist_rect[3] == 0:
            return image # no intersection
        src_offset = ( ist_rect[0]-x, ist_rect[1]-y )
        for ch in range(3):
            image[ist_rect[0]:ist_rect[0]+ist_rect[2], ist_rect[1]:ist_rect[1]+ist_rect[3], ch ]=\
                slice_data[src_offset[0]:src_offset[0]+ist_rect[2], src_offset[1]:src_offset[1]+ist_rect[3]]
        return image

def load_nifti_simple(nii_file):
    return nib.load(nii_file).get_fdata().astype('float32')

def lightbox(nii_file, n_rows, n_cols, save_path, slice_range=None, slice_step=None, view_axis='axial'):
    nii_data = load_nifti_simple(nii_file)

    assert view_axis in ['sagittal', 'coronal', 'axial'], '"view_axis" should be "sagittal", "coronal" or "axial"(default).'
    assert len(nii_data.shape) == 3, 'Only support 3D NIFTI data.'
    assert slice_range is None or isinstance(slice_range, (tuple, list)), 'Invalid "slice_range" setting.'
    assert len(save_path)>4 and save_path[-4:]=='.png', '"save_path" must ends with ".png".'

    if view_axis == 'sagittal':
        nii_slices = nii_data.shape[0]
        nii_data_tr = np.transpose(nii_data, [0,1,2])
        nii_data_tr = nii_data_tr[::-1,::-1,::-1]
    elif view_axis == 'coronal':
        nii_slices = nii_data.shape[1]
        nii_data_tr = np.transpose(nii_data, [1,0,2])
        nii_data_tr = nii_data_tr[::-1,::-1,::-1]
    elif view_axis == 'axial':
        nii_slices = nii_data.shape[2]
        nii_data_tr = np.transpose(nii_data, [2,0,1])

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

    # normalize nii intensity
    nii_data_tr = (nii_data_tr - nii_data_tr.min()) / (nii_data_tr.max()-nii_data_tr.min() + 0.00001)

    image = np.zeros([image_width, image_height, 3]) # RGB channel

    current_slice = slice_start
    for iy in range(n_rows):
        for ix in range(n_cols):
            if current_slice < nii_data_tr.shape[0]:
                # paste slice data
                _paste_slice(nii_data_tr[current_slice], image, slice_shape[0]*ix, slice_shape[1]*iy)
                # paste slice number
                slice_str = '%03d' % current_slice
                numbering_pos = (slice_shape[0]*ix+2, slice_shape[1]*iy+2)
                for ig in range(3): # max 3 digits
                    _paste_slice(glyph[int(slice_str[ig])], image, numbering_pos[0]+ig*6, numbering_pos[1])
            current_slice = int(current_slice + slice_step)

    imsave(save_path, np.transpose(image,[1,0,2]))
    





