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
    python abcrown.py --config exp_configs/my_configs/superclass.yaml --results_file "$results" --csv_name "instances_eps_5.csv" --load_model "$model"
done
# python abcrown.py --config exp_configs/my_configs/superclass.yaml --results_file "constraint_only_10_eps" --csv_name "instances_eps_5.csv" --load_model "../../Superclass/models/constraint_only_10_eps.pth"