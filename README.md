# AutoML TPOT

Simple introductory project to automatic machine learning with TPOT.

## Data

Unzip the contents of  `data/mnist-in-csv.zip` file inside the `data` folder.

## Installation

I recommend you use `virtualenv`. If you have it installed, create a new environment and activate it:

```bash
$> virtualenv -p python3 venv
$> source venv/bin/activate
```

After that, just install the requirements:

```bash
pip install -r requirements.txt
```

## Try It

You can fire up an automatic search like this:

```bash
python train.py
```

Keep in mind this is a heavy and long running process.