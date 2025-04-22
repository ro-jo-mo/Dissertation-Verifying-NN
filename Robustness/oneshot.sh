cd ..
cd alpha-beta-CROWN
cd complete_verifier

python abcrown.py --config exp_configs/my_configs/robustness_grey_base.yaml --epsilon 0.0196 --results_file "base_epsilon_20.pth.pkl" --load_model "../../Robustness/models/base_epsilon_20.pth"
python abcrown.py --config exp_configs/my_configs/robustness_grey_base.yaml --epsilon 0.0196 --results_file "base_epsilon_5.pth.pkl" --load_model "../../Robustness/models/base_epsilon_5.pth"