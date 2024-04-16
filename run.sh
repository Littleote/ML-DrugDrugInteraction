#! /bin/bash

BASEDIR=$(cat run.config | head -1 | tail -1)
PYTHON=$(cat run.config | head -2 | tail -1)

./corenlp-server.sh -quiet true -port 9000 -timeout 15000  &
sleep 1

# extract features
echo "Extracting features"
"$PYTHON" extract-features.py $BASEDIR/data/devel/ > devel.cod &
"$PYTHON" extract-features.py $BASEDIR/data/train/ | tee train.cod | cut -f4- > train.cod.cl

kill `cat ./tmp/corenlp-server.running`

# train model
echo "Training model"
"$PYTHON" train-sklearn.py model.joblib vectorizer.joblib < train.cod.cl
# run model
echo "Running model..."
"$PYTHON" predict-sklearn.py model.joblib vectorizer.joblib < devel.cod > devel.out
# evaluate results
echo "Evaluating results..."
"$PYTHON" evaluator.py DDI $BASEDIR/data/devel/ devel.out > devel.stats

