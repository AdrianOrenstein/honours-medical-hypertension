# Honours Medical Hypertension

Given free-text patient records, perform binary classification as to whether the agent has been diagnosed by a GP.

Results using a [balanced dataset](https://wandb.ai/adrianorenstein2/hypertension-project/reports/Hypertension-reproduce-results--VmlldzoyNDQ0OTg2?accessToken=0r6szww4emgau9d6i7rirzy3yxmtgdjri2mtns8ycr2l4i4ctwa207ytvlkdpkmh) and [imbalanced dataset](https://wandb.ai/adrianorenstein2/hypertension-project-unbalanced-test/reports/Hypertension-imbalanced-dataset--VmlldzoyNDU4MjYz?accessToken=cb93gkt4hiimy7ry4cl7e96t2snee0p09gx13zx18j10r2zucvcxxtple1pxj0mw) are available as WandB reports.

## Running experiments
Below are sections that outline various aspects of using this codebase.

### Development Environment
A [Makefile](https://opensource.com/article/18/8/what-how-makefile) is used for easy access to build the container, run interactive terminals and launch jupyterlab for analysis.
- `make build` builds the container.
- `make run` spawns a container and opens an interactive terminal.
- `make jupyter` spawns a container and opens jupyter lab, use the website token to open in your preferred browser.
- `make lint` runs linting and prettifies code inside the `src/` directory. 


### Logging Experiments
In `dockerfiles/Dockerfile` on the last line set your API key and run the docker container.

This project is setup to use [Weights & Biases](https://wandb.ai/site), obtain your API key from there.

### Preprocessing the data
Under `jupyter_notebooks/preprocess.ipynb` is a notebook that executes preprocessing.

The notebook first reads the patient, script items, reasons for prescription, pathology, immunisations, diagnosis encounters, and observation [Stata](https://pandas.pydata.org/docs/reference/api/pandas.read_stata.html) files. 

Preprocessing then drops unnecessary columns, and aggregates all patient data into a JSON file per patient. 
Each patient JSON file consists of an `id: int`, `data: JSON` and `label: int`. 

A data schema is formed, this is used to tokenise each field into a key and list of characters. 
The schema is saved for use during training. 

For every patient record, the record is assembled into a PatientData object and serialised to either the training or testing data directory.

### Running an experiment
Inside a docker container run:
```bash
python src/train.py --batch_size=32 --balanced_dataset=True --learning_rate=0.05

python src/train.py --batch_size=32 --balanced_dataset=False --learning_rate=0.05
```

#### The Model
In `src/experiments/resnet.py` a JSONKeyValueResnet_EmbeddingEncoder model is made (defined in `src/models/resnet_key_value.py`) which takes the JSON patient record, tokenises it ready for a dataloader.

The dataloader pads the batch to be a constant size, padding or cropping as needed.

Tokens are defined by the schema (our list of patient record fields) and all printable python characters. 
Unicode is not supported. 

The model outputs a 2 class multi-class output, this objective is trained with [CrossEntropyLoss](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html).


### Measuring model performance
In `jupyter_notebooks/analyse_performance.ipynb`. Load model checkpoint, run inference on the testing set, measure true positives and groundtruth labels, compute AUROC curve and calibrate threshold. 

### Measuring model GradCAM
In `jupyter_notebooks/analyse_gradcam.ipynb`. Load model checkpoint, compute gradient of class prediction w.r.t. patient record, render a heatmap of what parts of the input made the greatest change in confidence for a hypertension diagnosis. 