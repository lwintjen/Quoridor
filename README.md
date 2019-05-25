# QUORIDOR GAME - LEARNING BY REINFORCEMENT
This repository is used to develop an AI for the Quoridor Game. In this part of the project, we try to optimize the AI.

### The following updates have been added in the code :

* Activation functions :
	1. ReLU
	2. Leaky ReLU
	3. Hyperbolic tangent
	4. SWISH
* E-Greedy :
	1. Decreasing
	2. Bruit
* Soft-max 
* Modification of the GUI :
	1. Can choose the value of the decreasing E-greedy and the speed of the decay rate
	2. Can choose which gloutonne strategy and mix Soft-max and E-greedy
	3. Can choose the activation function
* Can now train the IA and compare it with more than 1 wall


### What do you need to run the program :

* Numpy,os,sys libraries
* Python3
* IA_partie2.py
* partie4.py
* utils_partie2.py

### If you want to use the program without a GUI :

* Get the file tournoi.py and you will need : 
	1. IA_partie2.py 
	2. utils_partie2.py 
	3. partie4.py
* Open your terminal and use the cmd : 
	1. python3 tournoi.py nb_of_games_to_train_the_AI name_of_new_file.npz
	2. i.e : python3 tournoi.py 10000 my_new_AI.npz

Now you know everything you need to modify, use and enjoy this program ! :ok_hand:

This project has been made as a Year Project in my Computer Science freshman year at the University of Brussels. Credits to : Gwenaël Joret, Charlotte Nachtegael, Arnaud Pollaris, Cédric Ternon, Jérôme De Boeck.
