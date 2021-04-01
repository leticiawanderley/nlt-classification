# Negative language transfer classification

This repository hosts the code used to process and analyse the [negative language transfer dataset](https://github.com/EdTeKLA/LanguageTransfer). This implementation trains two machine learning classification models to predict whether English learner errors are related to negative language transfer. Parameter tuning is used to select the best features to model the task. After training and testing, the models' performances on the task are compared and contrasted.

* The scripts inside the repository are implemented in Python 3.7

## Data
The data used in this project comes from the the [learner English negative language transfer dataset](https://github.com/EdTeKLA/LanguageTransfer). It contains Chinese native speakers English errors annotated with information about their relation to negative language transfer. To have access to this dataset, you need to follow the guidance provided in its repository.

## Processing
The negative language transfer annotated data was preprocessed to extract the part-of-speech and dependency tags associated with the learner errors and their surrounding. The [pre_process_data.py](pre_process_data.py) script does this syntactic feature extraction and creates a new processed datafile. To run the preprocessing script replace the file name on line 80 with the path to the negative language transfer annotated dataset.

The [process_data.py](process_data.py) script should be ran after data preprocessing. It creates dummy and encoded versions of the data's categorical features. Then, it splits the processed dataset into train and test files.

## Tuning
To select the best parameters for the random forest model used in this classification task, a systematic search over the parameter space was performed. The parameter combinations creation can be found in the [create_input_files.py](create_input_files.py) script. The parameter tuning results are available in [data/results_tuning.csv](data/results_tuning.csv).

## Analysis
The training and testing of the final classification models can be run via a jupyter notebook. Run `jupyter notebook` on the repository's root to access the [NLT_classification.ipynb](NLT_classification.ipynb) interactive analysis notebook.