-- Todo
-- Recompile marabou with gorobi
-- Write robustness property in vehicle
-- Create idx dataset?? 

type Image = Tensor Rat [32, 32]
type Output = Vector Rat 43
type Label = Index 43
@network
classifier : Image -> Output

correctlyClassifies : Image -> Label -> Bool
correctlyClassifies img label = forall i . 
    i != label => classifier img ! label > classifier img ! i

@parameter(infer=True)
n : Nat

@dataset
images : Vector Image n

@dataset
labels : Vector Label n

@property
accuracy : Vector Bool n
accuracy = foreach i . 
    correctlyClassifies (images ! i) (labels ! i)