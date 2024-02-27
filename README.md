# LorcanaSiamese
Lorcana Card Recognition Iteration 3 implementing a Siamese Model

This is the third attempt to create a computer vision program that can detect which Lorcana trading card is shown in an image.

The first iteration was a classification model in C+, which after trouble with the VM i was creating it in, I shift to a similar model in Python on my local machine. Iteration 2 is the Python model, which was attempting to classify within 432 classes (as at this time there are 432 different lorcana cards), but after testing and firther research found that a Siamese model would be more effective as the amount of classes scaled (with ~200 new cards being released every 3 months)

So far this iteration consists of the pair generation, building and training of the Siamese model. As of 2/18/24 The model is training very poorly with an accuracy ~67. I am now begining to investiagate how to increase that accuracy and increasing efficiency of the program.

Further research indicates utilizing triplet loss may lead to better accuracy. Working on implementation
