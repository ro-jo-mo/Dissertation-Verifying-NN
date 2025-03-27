cd ..
cd alpha-beta-CROWN
cd complete_verifier

models="../../Superclass/models/"
for model in "$models"*.pth; do
    file=$(basename "$model")
    echo "File: $file"
    echo "Path: $model"
    results=$file"_superclass.pkl"
    echo "Results: $results"
    python abcrown.py --config exp_configs/my_configs/superclass.yaml --results_file "$results" --load_model "$model"
done
