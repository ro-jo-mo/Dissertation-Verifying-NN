type Image = Tensor Rat [32, 32] -- Represents an image
type Group = Index 3 -- Represents one of the six groups
type Label = Index 20 -- Represents a label, one of the 43 signs
type Prediction = Vector Rat 20 -- Represents the output of the model
-- for groups ! i ! j, this evaluates True when label j is a member of group i
groups : Tensor Bool [6,43]
groups = [
    [False, False, False, False, False, False, True, False, False, True, False, False, False, False, False, False, False, False, True, True],
    [True, True, True, True, True, True, False, True, True, False, False, False, False, False, False, False, False, False, False, False],
    [False, False, False, False, False, False, False, False, False, False, True, True, True, True, True, True, True, True, False, False]
]
    
memberOfGroup : Label -> Group -> Bool
memberOfGroup label group = groups ! group ! label

-- Normalisation is baked into the dataset 


upperBound = 3.044394618834081
lowerBound = -1.4399103139013452
validImage : Image -> Bool
validImage x = forall i j . lowerBound <= x ! i ! j <= upperBound


@network
classifier : Image -> Prediction

-- The classifier advises that input image `x` has label `i` if the score
-- for label `i` is greater than the score of any other label `j`.

advises : Prediction -> Label -> Group -> Bool
advises pred label group = forall i .
    not memberOfGroup i group => pred ! label >= pred ! i

correctlyGroups : Prediction -> Group -> Bool
correctlyGroups pred group = forall label . -- If this label is a member of the group, check it is predicted higher than labels outside the group
    memberOfGroup label group => advises pred label group



--------------------------------------------------------------------------------
-- Definition of robustness around a point

-- First we define the parameter `epsilon` that will represent the radius of the
-- ball that we want the network to be robust in. Note that we declare this as
-- a parameter which allows the value of `epsilon` to be specified at compile
-- time rather than be fixed in the specification.
@parameter
epsilon : Rat

-- Next we define what it means for an image `x` to be in a ball of
-- size epsilon around 0.
boundedByEpsilon : Image -> Bool
boundedByEpsilon x = forall i j . -epsilon <= x ! i ! j <= epsilon

-- We now define what it means for the network to be robust around an image `x`
-- that should be classified as `y`. Namely, that for any perturbation no greater
-- than epsilon then if the perturbed image is still a valid image then the
-- network should still advise label `y` for the perturbed version of `x`.
robustAround : Image -> Group -> Bool
robustAround image group = forall perturbation .
    let perturbedImage = image - perturbation in
    let predictions = classifier perturbedImage in
    boundedByEpsilon perturbation and validImage perturbedImage =>
        correctlyGroups predictions group

--------------------------------------------------------------------------------
-- Robustness with respect to a dataset

-- We only really care about the network being robust on the set of images it
-- will encounter. Indeed it is much more challenging to expect the network
-- to be robust around all possible images. After all most images will be just
-- be random noise.

-- Unfortunately we can't characterise the set of "reasonable" input images.
-- Instead we approximate it using the training dataset, and ask that the
-- network is robust around images in the training dataset.

-- We first specify parameter `n` the size of the training dataset. Unlike
-- the earlier parameter `epsilon`, we set the `infer` option of the
-- parameter `n` to 'True'. This means that it does not need to be provided
--  manually but instead will be automatically inferred by the compiler.
-- In this case it will be inferred from the training datasets.
@parameter(infer=True)
n : Nat

-- We next declare two datasets, the training images and the corresponding
-- training labels. Note that we use the previously declared parameter `n`
-- to enforce that they are the same size.
@dataset
images : Vector Image n
-- Which group this image is in
@dataset
labels : Vector Group n

-- For all classes in this group, 

-- We then say that the network is robust if it is robust around every pair
-- of input images and output labels. Note the use of the `foreach`
-- keyword when quantifying over the index `i` in the dataset. Whereas `forall`
-- would return a single `Bool`, `foreach` constructs a `Vector` of booleans,
-- ensuring that Vehicle will report on the verification status of each image in
-- the dataset separately. If `forall` was omitted, Vehicle would only
-- report if the network was robust around *every* image in the dataset, a
-- state of affairs which is unlikely to be True.
@property
robust : Vector Bool n
robust = foreach i . robustAround (images ! i) (labels ! i)
