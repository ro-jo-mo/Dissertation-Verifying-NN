cd ..
cd alpha-beta-CROWN
cd complete_verifier


python abcrown.py --config exp_configs/my_configs/superclass.yaml --results_file "5_eps_constraint_only.pkl" --csv_name "instances_eps_5.csv" --load_model "../../Superclass/models/constraint_only_10_eps.pth"
python abcrown.py --config exp_configs/my_configs/superclass.yaml --results_file "5_eps_hybrid.pkl" --csv_name "instances_eps_5.csv" --load_model "../../Superclass/models/hybrid_10_eps.pth"
python abcrown.py --config exp_configs/my_configs/superclass.yaml --results_file "5_eps_base.pkl" --csv_name "instances_eps_5.csv" --load_model "../../Superclass/models/superclass_base.pth"

python abcrown.py --config exp_configs/my_configs/superclass.yaml --results_file "1_eps_constraint_only.pkl" --csv_name "instances_eps_1.csv" --load_model "../../Superclass/models/constraint_only_10_eps.pth"
python abcrown.py --config exp_configs/my_configs/superclass.yaml --results_file "1_eps_hybrid.pkl" --csv_name "instances_eps_1.csv" --load_model "../../Superclass/models/hybrid_10_eps.pth"

python abcrown.py --config exp_configs/my_configs/superclass.yaml --results_file "3_eps_constraint_only.pkl" --csv_name "instances_eps_3.csv" --load_model "../../Superclass/models/constraint_only_10_eps.pth"
python abcrown.py --config exp_configs/my_configs/superclass.yaml --results_file "3_eps_hybrid.pkl" --csv_name "instances_eps_3.csv" --load_model "../../Superclass/models/hybrid_10_eps.pth"

python abcrown.py --config exp_configs/my_configs/superclass.yaml --results_file "7_eps_constraint_only.pkl" --csv_name "instances_eps_7.csv" --load_model "../../Superclass/models/constraint_only_10_eps.pth"
python abcrown.py --config exp_configs/my_configs/superclass.yaml --results_file "7_eps_hybrid.pkl" --csv_name "instances_eps_7.csv" --load_model "../../Superclass/models/hybrid_10_eps.pth"

python abcrown.py --config exp_configs/my_configs/superclass.yaml --results_file "10_eps_constraint_only.pkl" --csv_name "instances_eps_10.csv" --load_model "../../Superclass/models/constraint_only_10_eps.pth"
python abcrown.py --config exp_configs/my_configs/superclass.yaml --results_file "10_eps_hybrid.pkl" --csv_name "instances_eps_10.csv" --load_model "../../Superclass/models/hybrid_10_eps.pth"

