# RepsCounter

RepsCounter is a project I thought of over the summer when I injured my shoulder and could not lift heavy weights, but still wanted a way to exercise. The only way to do so would be through body weight exercises, but I thought of them too tedious. My first idea was to instead count number of squat repititons, but I thought data collection would be somewhat difficult since all landmarks have to be present (though I do plan to do it in the future), so I instead opted for number of repetitions of the hand gripper exercise.


### How it works
Run fingertracking.py, and make sure to see the outline of your hand on the screen, everytime you perform a repeititon of the hand gripper exercise, it should print the number of repetitions done so far to the console, press r to reset the count.

### Quirks
Unfortunately the model is somewhat far from perfect. First off, if I were to position my hand down (imagine a basketball player's hand after they shot the basketball), and try performing the exercise, it does not count it. Secondly, if I move hand such that my fingers are positioned above my palm, and move them so that they are below my palm, and back up, this counts it as one repetiton, even though it is not.
What I plan to do to improve this is also add in data of the angle between joints of the finger to the model as that is a better indication of a close/open position rather than the normalized position of parts of the hand.

### Difficulties

First off was getting an environment where all of OpenCV, Tensorflow, and MediaPipe to all work together. I could only find Python 3.8 for this, thus Python 3.8 is needed to run all this. 
Secondly, loading the Tensorflow model into the main script seemed to have a problem. Tensorflow automatically switches from using `tf.keras.optimizers.Adam()` to `tf.keras.optimizers.legacy.Adam()` since the machine I am using runs on Apple Silicon. This led to a problem loading in the model from the .keras file as the legacy version as the Adam object had no attribute build. I believe this issue was fixed in Tensorflow 2.15 however I was unable to update to this version in the virtual environment I was using, thus I decided to switch entirely to PyTorch.
