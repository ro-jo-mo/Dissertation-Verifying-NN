cd ..
cd alpha-beta-CROWN
cd complete_verifier


python abcrown.py --config exp_configs/my_configs/robustness_grey_base.yaml --epsilon 0.0784 --results_file "robust_10_eps_20.pkl" --load_model "../../Robustness/models/base_epsilon_10.pth"
python abcrown.py --config exp_configs/my_configs/robustness_grey_base.yaml --epsilon 0.0039 --results_file "robust_10_eps_1.pkl" --load_model "../../Robustness/models/base_epsilon_10.pth"
python abcrown.py --config exp_configs/my_configs/robustness_grey_base.yaml --epsilon 0.0078 --results_file "robust_10_eps_2.pkl" --load_model "../../Robustness/models/base_epsilon_10.pth"
python abcrown.py --config exp_configs/my_configs/robustness_grey_base.yaml --epsilon 0.0118 --results_file "robust_10_eps_3.pkl" --load_model "../../Robustness/models/base_epsilon_10.pth"
python abcrown.py --config exp_configs/my_configs/robustness_grey_base.yaml --epsilon 0.0157 --results_file "robust_10_eps_4.pkl" --load_model "../../Robustness/models/base_epsilon_10.pth"
python abcrown.py --config exp_configs/my_configs/robustness_grey_base.yaml --epsilon 0.0294 --results_file "robust_10_eps_7.5.pkl" --load_model "../../Robustness/models/base_epsilon_10.pth"
python abcrown.py --config exp_configs/my_configs/robustness_grey_base.yaml --epsilon 0.0392 --results_file "robust_10_eps_10.pkl" --load_model "../../Robustness/models/base_epsilon_10.pth"
python abcrown.py --config exp_configs/my_configs/robustness_grey_base.yaml --epsilon 0.0490 --results_file "robust_10_eps_12.5.pkl" --load_model "../../Robustness/models/base_epsilon_10.pth"
python abcrown.py --config exp_configs/my_configs/robustness_grey_base.yaml --epsilon 0.0588 --results_file "robust_10_eps_15.pkl" --load_model "../../Robustness/models/base_epsilon_10.pth"
python abcrown.py --config exp_configs/my_configs/robustness_grey_base.yaml --epsilon 0.0686 --results_file "robust_10_eps_17.5.pkl" --load_model "../../Robustness/models/base_epsilon_10.pth"
