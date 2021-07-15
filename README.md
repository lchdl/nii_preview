# nii_preview
A simple utility to visualize 3D NIFTI image based on Python3.

## How to install
**all you need is the "nll_preview.py" located in this repo.**
but also, you need to install the following Python packages (ignore if you already installed):
> 1) **matplotlib** (any version should work)
> 2) **numpy** (any version should work)
> 3) **nibabel** (used for loading "\*.nii" or "\*.nii.gz" files, recommend version >=3.2.1, but lower version is OK I guess)

## How to use
here is an example of how to use it

> from nii_preview import lightbox
> # Generate a 4x5 image preview, view axis is set to 'axial', 
> # file will be saved to 'axial_preview.png'.
> lightbox('/location/to/your/sample/file.nii.gz', 4, 5,'axial_preview.png', view_axis='axial')
> # However, you can set "view_axis" to "coronal" or "sagittal" 
> lightbox('/location/to/your/sample/file.nii.gz', 4, 5,'axial_preview.png', view_axis='coronal')
> lightbox('/location/to/your/sample/file.nii.gz', 4, 5,'axial_preview.png', view_axis='sagittal')
