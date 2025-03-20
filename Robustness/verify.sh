cd ..
cd alpha-beta-CROWN
cd complete_verifier

models="/home/rob/code/Project/Robustness/models/"
for model in "$models"*.pth; do
    file=$(basename "$model")
    echo "File: $file"
    echo "Path: $model"
    results=$file".pkl"
    echo "Results: $results"
    python abcrown.py --config exp_configs/my_configs/robustness_grey_base.yaml --epsilon 0.0196 --results_file "$results" --load_model "$model"
done
# epsilon = 5 / 255 
# python abcrown.py --config exp_configs/my_configs/robustness_grey_base.yaml --epsilon 0.0196 --resultssss --load_model "/home/rob/code/Project/Robustness/models/base_epsilon_0.000.pth"
