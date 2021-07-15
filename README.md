# nii_preview
A simple utility to visualize 3D NIFTI image based on Python3.

## How to install
**All you need is the "nll_preview.py" located in this repo.**

Copy **"nll_preview.py"** into your project directory and everything is done :).
But also, you need to install the following Python packages (ignore if you already installed):
> 1) **matplotlib** (any version should work)
> 2) **numpy** (any version should work)
> 3) **nibabel** (used for loading "\*.nii" or "\*.nii.gz" files, recommend version >=3.2.1, but lower version is OK I guess)

## How to use
here is an example of how to use it

<p align="left">
  <img 
       src="https://github.com/lchdl/nii_preview/blob/main/how_to_use.png"
       width="800"
  />
</p>

## Output Example
Below is an example of using "lightbox()" function to generate a preview of a 3D NIFTI file. "view_axis" is set to "coronal".
Note that slice number is also indicated in the upper left corner. I carefully adjusted the behaviour of "lightbox()" to
ensure that the visualization result is the same with that of [MIPAV](https://mipav.cit.nih.gov/) and 
[ITK-SNAP](http://www.itksnap.org/pmwiki/pmwiki.php).

<p align="left">
  <img 
       src="https://github.com/lchdl/nii_preview/blob/main/lightbox_coronal.png"
       width="800"
  />
</p>
