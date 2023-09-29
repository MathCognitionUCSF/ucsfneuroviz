# %%
import os
import nibabel as nib
import plotly
import numpy as np
from dipy.io.streamline import load_tractogram
from dipy.io.stateful_tractogram import Space
from AFQ.utils.streamlines import SegmentedSFT
from AFQ.viz.plotly_backend import visualize_bundles, visualize_volume


working_dir = '/Volumes/language/dyslexia/browser_development/qsiprep'
tract_base_path = os.path.join(working_dir, 'sub-34429_recon/ses-20230317/dwi/sub-34429_ses-20230317_space-T1w_desc-preproc/clean_bundles/')
