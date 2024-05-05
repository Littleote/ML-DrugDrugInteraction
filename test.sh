#! /bin/bash

BASEDIR=$(cat run.config | head -1 | tail -1)
PYTHON=$(cat run.config | head -2 | tail -1)

./corenlp-server.sh -quiet true -port 9000 -timeout 15000  &
sleep 1

# extract features
echo "Extracting features"
"$PYTHON" extract_features.py $BASEDIR/data/test/ > test.cod &
"$PYTHON" extract_features.py $BASEDIR/data/devel/ > devel.cod &
"$PYTHON" extract_features.py $BASEDIR/data/train/ | tee train.cod | cut -f4- > train.cod.cl

kill `cat ./tmp/corenlp-server.running`

# train model
echo "Training model"
"$PYTHON" train_sklearn.py model.joblib vectorizer.joblib < train.cod.cl
# run model
echo "Running model..."
"$PYTHON" predict_sklearn.py model.joblib vectorizer.joblib < test.cod > test.out
"$PYTHON" predict_sklearn.py model.joblib vectorizer.joblib < devel.cod > devel.out
"$PYTHON" predict_sklearn.py model.joblib vectorizer.joblib < train.cod > train.out
# evaluate results
echo "Evaluating results..."
"$PYTHON" evaluator.py DDI $BASEDIR/data/test/ test.out CM-test.png > test.stats
"$PYTHON" evaluator.py DDI $BASEDIR/data/devel/ devel.out CM-devel.png > devel.stats
"$PYTHON" evaluator.py DDI $BASEDIR/data/train/ train.out CM-train.png > train.stats

