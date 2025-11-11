# Verification of Roadsign Detection
My dissertation project for BSc at Southampton.

This project focuses on formally guaranteeing the robustness of roadsign detection (small alterations to images don't affect classifications), through using verifiers to compute guarantees. 
Moreover I work on training more robust models through adversarial training techniques. 
A large portion of this project was spent optimising the model such that verification was feasible, as this is a very expensive process.
Additionally a more complex "common sense" superclass property is trained for and verified, where similar signs are grouped together (speed limits, warning signs etc), and misclassifications within a class are allowed.

# Report
My report can be found [here](https://github.com/ro-jo-mo/Dissertation-Verifying-NN/blob/main/report.pdf).
