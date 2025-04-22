cd ..
cd alpha-beta-CROWN
cd complete_verifier

models="../../Robustness/models/"
for model in "$models"*.pth; do
    file=$(basename "$model")
    echo "File: $file"
    echo "Path: $model"
    results=$file".pkl"
    echo "Results: $results"
    echo "python abcrown.py --config exp_configs/my_configs/robustness_grey_base.yaml --epsilon 0.0196 --results_file \"$results\" --load_model \"$model\""
    # python abcrown.py --config exp_configs/my_configs/robustness_grey_base.yaml --epsilon 0.0196 --results_file "$results" --load_model "$model"
done
