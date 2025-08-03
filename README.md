# ODATQA-MDC

This repository contains resources and datasets for the **ODATQA-MDC** project, designed for **Open-Domain Ambiguous Tabular Question Answering**.

To facilitate the reproducibility of our experiments in **MAIC-TQA**, we provide a unified launch script that initializes all modules. This script is located in the `scripts` directory. Before running the script, please make sure to follow the steps in the [Getting Started](#getting-started) section to set up the required environment.

If you have any questions about this project, feel free to open an issue (after the code is made public, following the peer review process). We will respond accordingly.


![Overview of the MAIC-TQA pipeline](figures/Workflow.png)

---

## 📁 Directory Structure

```
.
├── datasets                    # All QA datasets and tabular data
│   ├── README.md
│   ├── table_captions
│   ├── tables
│   ├── test.json
│   ├── train.json
│   └── validation.json
├── figures
├── MAIC-TQA                    # Source code for MAIC-TQA
│   ├── 01_BERT_finetune
│   ├── 02_SLU_module
│   ├── 03_Scope_Validator_Agent
│   ├── 04_Table_Retrieval_Agent
│   ├── 05_SQL_Generation_and_Validation_Agent
│   └── prompts
├── README.md
├── requirements.txt            # Python dependencies for the experiments
└── scripts
    └── start.sh                # Unified script to launch MAIC-TQA experiments

13 directories, 7 files
```

---

## 🚀 Getting Started

### Step 1: Set Up Python Environment

All Python dependencies are listed in `requirements.txt`. Please ensure you have **Conda** installed, and then run the following commands to create a virtual environment and install the necessary packages:

```bash
# Create a conda environment named 'odatqa'
conda create -n odatqa python==3.10.0

# Activate the environment
conda activate odatqa

# Install required packages
pip install -r requirements.txt
```

---

### Step 2: Deploy and Initialize the Database

We recommend using **Docker** to deploy a PostgreSQL database. Once Docker is installed, run the following command to create a PostgreSQL container:

```bash
docker run -id \
  --name=re-postgres \
  -v ./data:/var/lib/postgresql/data \
  -p 25432:5432 \
  -e POSTGRES_PASSWORD='odatqa123456' \
  -e POSTGRES_USER='odatqa' \
  -e LANG=C.UTF-8 \
  --restart=always \
  postgres:alpine
```

This will create an **empty database**. To build a complete TableQA environment, you'll need to import the predefined tables using the script in `datasets/tables/import_table.ipynb`.

Please follow these steps:

1. **Unzip the data files**:

   ```bash
   tar -xvf datasets/tables/tables.tar
   ```

2. **Open and execute** the Jupyter notebook:

   ```bash
   datasets/tables/import_table.ipynb
   ```

3. ⚠️ **Important**: Before execution, be sure to update the database IP address in the notebook to point to your local PostgreSQL instance.

---

### Step 3: Launch the Full Pipeline (Required)

After configuring the Python environment and initializing the database, you can start the MAIC-TQA pipeline using the unified launch script:

1. **Edit** the file `scripts/start.sh` to configure your model service `baseURL` and `API Key`.

2. **Run the script** with:

   ```bash
   cd scripts/
   bash start.sh your_model_name 10
   ```

* **Parameters**:

  * `your_model_name`: Your preferred model's name.
  * `max_parallel_requests`: The maximum number of concurrent requests your model API can handle (e.g., 10).

---