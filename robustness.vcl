-- vehicle verify --specification robustness.vcl --network classifier:models/model.onnx --dataset images:dataset/images.idx --dataset labels:dataset/labels.idx --verifier Marabou
type Image = Tensor Int [32, 32]
type Output = Vector Rat 43

@network
classifier : Image -> Output


-- How much perturbation can occur on the image
@parameter
epsilon : Rat

-- Ensures a given perturbation is in the bound of the ball
-- Using L-inf norm
inEpsilonBall : Image -> Bool 
inEpsilonBall perturbations = forall i j . -epsilon <= perturbations ! i ! j <= epsilon

-- Ensures all pixel values are within the boundaries for colour 
validImage : Image -> Bool
validImage img = forall i j k . 0 <= x ! i ! j <= 1

-- Standard Robustness 
--standardRobustness : Image -> Bool
--standardRobustness image = forall perturbation . 
--    let perturbedImage = image + perturbation in
--    inEpsilonBall perturbation and validImage perturbedImage
--    => 
    

advises : Image -> Label -> Bool
advise img label = forall i .
    i != label => classifier img ! label > classifier img ! i


robustAround : Image -> Label -> Bool
robustAround image label = forall perturbation .
    let perturbedImage = image + perturbation in
    inEpsilonBall perturbation and validImage perturbedImage =>   
    advises perturbedImage label


@parameter(infer=True)
n : Nat

-- We next declare two datasets, the training images and the corresponding
-- training labels. Note that we use the previously declared parameter `n`
-- to enforce that they are the same size.
@dataset
trainingImages : Vector Image n

@dataset
trainingLabels : Vector Label n

@property
robust : Vector Bool n
robust = foreach i . robustAround (trainingImages ! i) (trainingLabels ! i)
