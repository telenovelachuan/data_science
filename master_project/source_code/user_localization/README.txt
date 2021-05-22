Two files are included in the folder:
1. preprocessing.py: Data loading, preprocessing and dimensionality reduction.
2. localization.py: Training base learner model, implementing active learning pipeline and saving results.

Run preprocessing.py first so that an intermediate file is generated. Then run localization.py for getting all results. Both files should be run at the same directory as the raw data files(file_xxx.hdf5).