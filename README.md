# Need For Speed_v1
Hi, This is Gautam.
This is the same as shown in a tutorial with no mods of my own. 

Run Need For Speed Most Wanted (old version) in a windowed mode with dimensions 50, 50, 800, 500.

Run Python scripts as numbered. Make sure your python IDLE/CMD is only on the left side of your screen.

We run main.py to capture the screen and input data. Game needs to be open, and you need to play the game(how you want the neural network to play).

We use the captured data as training data for our convolutional neural network.

We have a new file - training_data.npy

We run balance_data.py to balance the data to make sure we don't over fit. Game need not be open.

We have a new file - training_data_v2.npy

We run train_model.py to train our model using training_data_v2.npy Game need not be open.

We run test_model.py to test our trained model. Game needs to be open and selected after running test_model.py. Wait for the countdown, and your car should drive automatically.

Feel free to pull request or make changes according to your convinience. You can even tweak the codes such that it runs for any game you like.

Thanks.
