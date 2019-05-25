"""
Auteur : Loris Wintjens avec l'aide de Charlotte Nachtegael, sur base du code de Gwenaël Joret et Arnaud Pollaris
Titre : Optimisation d'une IA - INFO-F106
Date : 7/04/19
"""

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from utils_partie2 import *
import time, random
import numpy as np

IMG_WHITE = QImage("./images/White_dot.png")
IMG_BLACK = QImage("./images/Black_dot.png")

MSG_MOVEMENT = 'Choisissez un mouvement (flèches pour bouger, w pour poser un mur, q pour quitter)'
MSG_DIAGONAL =  'Saut diagonal, choisissez la destination'
END_GAME = "FINI !"

class App(QMainWindow):
    
    def __init__(self):
        super().__init__()

        self.title = 'Quoridor INFOF106'
        self.left = 10
        self.top = 10
        self.width = 1400
        self.height = 800
        self.initUI()


    def initUI(self):
        self.setWindowTitle(self.title)

        self.centralwidget = QWidget()

        # main box
        self.hb = QHBoxLayout()

        # options and parameters box
        vb = QVBoxLayout()
        grid = QGridLayout()


        file_to_save = QLabel('Nom de fichier à sauvegarder')
        self.SaveEdit = QLineEdit()
        self.SaveButton = QPushButton()
        self.SaveButton.setText("Save file")
        self.SaveButton.clicked.connect(self.save)

        file_to_load = QLabel('Nom de fichier à charger')
        self.LoadEdit = QLineEdit()
        self.LoadButton = QPushButton()
        self.LoadButton.setText("Load file")
        self.LoadButton.clicked.connect(self.load)

        SizeLabel = QLabel("Taille du plateau")
        self.Size = QSpinBox()
        self.Size.setMinimum(5)
        self.Size.setMaximum(9)
        self.Size.setSingleStep(2)
        self.Size.setValue(5)

        WallLabel = QLabel("Nombre de murs")
        self.Wall = QSpinBox()
        self.Wall.setMinimum(0)
        self.Wall.setMaximum(10)
        self.Wall.setSingleStep(1)
        self.Wall.setValue(1)

        EpsilonLabel = QLabel("Valeur initiale epsilon")
        self.Epsilon = QDoubleSpinBox()
        self.Epsilon.setMinimum(0)
        self.Epsilon.setMaximum(0.9)
        self.Epsilon.setSingleStep(0.1)
        self.Epsilon.setValue(0.9)

        endValueEpsilon = QLabel("Valeur de fin epsilon (mettre même valeur que eps initial si pas de eps décroissant)")
        self.endEpsilon = QDoubleSpinBox()
        self.endEpsilon.setMinimum(0)
        self.endEpsilon.setMaximum(0.9)
        self.endEpsilon.setSingleStep(0.1)
        self.endEpsilon.setValue(0.1)

        decayRateEpsilonLabel = QLabel("Vitesse de décroissance eps")
        self.decayrateEpsilon = QDoubleSpinBox()
        self.decayrateEpsilon.setMinimum(0)
        self.decayrateEpsilon.setMaximum(0.9)
        self.decayrateEpsilon.setSingleStep(0.01)
        self.decayrateEpsilon.setValue(0.1)

        AlphaLabel = QLabel("Learning rate")
        self.Alpha = QDoubleSpinBox()
        self.Alpha.setMinimum(0)
        self.Alpha.setMaximum(0.9)
        self.Alpha.setSingleStep(0.1)
        self.Alpha.setValue(0.4)

        LambdaLabel = QLabel("Lambda for TD")
        self.Lambda = QDoubleSpinBox()
        self.Lambda.setMinimum(0)
        self.Lambda.setMaximum(0.9)
        self.Lambda.setSingleStep(0.1)
        self.Lambda.setValue(0.9)

        LearningStrategy= QLabel('IA strategy')
        self.LScb = QComboBox()
        self.LScb.addItem("Q-learning")
        self.LScb.addItem("TD-lambda")

        GloutonneStrategy = QLabel('Stratégie gloutonne')
        self.GScb = QComboBox()
        self.GScb.addItem("E-greedy")
        self.GScb.addItem("E-greedy avec bruit")
        self.GScb.addItem("Soft-max")
        self.GScb.addItem("Soft-max et E-greedy")
        self.GScb.addItem("Soft-max et E-greedy avec bruit")
        self.GScb.currentIndexChanged.connect(self.modifyGScb)
        self.GScb.currentIndexChanged.connect(self.modifyBruit)
        self.GScb.currentIndexChanged.connect(self.modifyGScbBruit)


        ActivationFunction = QLabel("Fonction d'activation")
        self.activate = QComboBox()
        self.activate.addItem("ReLU")
        self.activate.addItem("Sigmoid")
        self.activate.addItem("Tangente hyperbolique")
        self.activate.addItem("Leaky ReLU")
        self.activate.addItem("SWISH")



        LabelOne = QLabel('Player 1')
        self.PlayerOne = QComboBox()
        self.PlayerOne.addItem("IA")
        self.PlayerOne.addItem("Human")

        LabelTwo = QLabel('Player 2')
        self.PlayerTwo = QComboBox()
        self.PlayerTwo.addItem("IA")
        self.PlayerTwo.addItem("Human")

        number_train = QLabel("Nombre de tours pour entraîner/comparer l'IA")
        self.TrainNumber = QSpinBox()
        self.TrainNumber.setMinimum(100)
        self.TrainNumber.setMaximum(100000000)
        self.TrainNumber.setSingleStep(1000)
        self.TrainNumber.setValue(1000)

        number_NN = QLabel("Nombre de neurones")
        self.NumberNN = QSpinBox()
        self.NumberNN.setMinimum(10)
        self.NumberNN.setMaximum(100)
        self.NumberNN.setSingleStep(1)
        self.NumberNN.setValue(40)


        self.TrainButton = QPushButton()
        self.TrainButton.setText("Train the IA")
        self.TrainButton.clicked.connect(self.launch_train)

        self.CreateButton = QPushButton()
        self.CreateButton.setText("Create New IA")
        self.CreateButton.clicked.connect(self.create_new)

        self.PlayButton = QPushButton()
        self.PlayButton.setText("Play with current parameters")
        self.PlayButton.clicked.connect(self.launch_game)

        grid.addWidget(file_to_save, 1, 0)
        grid.addWidget(self.SaveEdit, 1, 1)
        grid.addWidget(self.SaveButton, 1, 2)

        grid.addWidget(file_to_load, 2, 0)
        grid.addWidget(self.LoadEdit, 2, 1)
        grid.addWidget(self.LoadButton, 2, 2)

        grid.addWidget(SizeLabel,3,0)
        grid.addWidget(self.Size,3,1)

        grid.addWidget(WallLabel,4,0)
        grid.addWidget(self.Wall,4,1)

        grid.addWidget(EpsilonLabel, 5, 0)
        grid.addWidget(self.Epsilon, 5, 1)

        grid.addWidget(endValueEpsilon,6,0)
        grid.addWidget(self.endEpsilon,6,1)

        grid.addWidget(decayRateEpsilonLabel,7,0)
        grid.addWidget(self.decayrateEpsilon,7,1)

        grid.addWidget(AlphaLabel, 8, 0)
        grid.addWidget(self.Alpha, 8, 1)

        grid.addWidget(LambdaLabel, 9, 0)
        grid.addWidget(self.Lambda, 9, 1)

        grid.addWidget(number_train, 10, 0)
        grid.addWidget(self.TrainNumber, 10, 1)

        grid.addWidget(number_NN, 11, 0)
        grid.addWidget(self.NumberNN, 11, 1)

        grid.addWidget(LearningStrategy, 12, 0)
        grid.addWidget(self.LScb, 12, 1)

        grid.addWidget(GloutonneStrategy, 13, 0)
        grid.addWidget(self.GScb,13,1)

        grid.addWidget(ActivationFunction, 14, 0)
        grid.addWidget(self.activate,14,1)

        grid.addWidget(LabelOne, 15, 0)
        grid.addWidget(self.PlayerOne, 15, 1)

        grid.addWidget(LabelTwo, 16, 0)
        grid.addWidget(self.PlayerTwo, 16, 1)

        grid.addWidget(self.CreateButton, 17, 0)
        grid.addWidget(self.TrainButton, 17, 1)
        grid.addWidget(self.PlayButton, 17, 2)



        vb.addLayout(grid)
        vb.addStretch(1)

        self.board = Grid_Board()
        self.grid_board = QGridLayout()
        self.grid_board.setSpacing(0)
        self.draw_map()
        self.board.setLayout(self.grid_board)
        self.board.moved.connect(self.redraw)
        self.board.exit.connect(self.reset_all)

        self.vb_board = QVBoxLayout()

        self.textTurn = QLabel()
        self.textGame = QLabel()

        self.vb_board.addStretch(1)
        self.vb_board.addWidget(self.board)
        self.vb_board.addWidget(self.textTurn)
        self.vb_board.addWidget(self.textGame)
        self.vb_board.addStretch(1)
        self.hb.addLayout(self.vb_board)
        self.hb.addLayout(vb)

        self.centralwidget.setLayout(self.hb)

        self.setCentralWidget(self.centralwidget)

        self.setGeometry(self.left, self.top, self.width, self.height)

        self.show()

    def save(self):
        """
        Save the AI in a given filename
        """
        if self.board.NN is None:
            QMessageBox.about(self, "Error", "Il faut d'abord créer ou charger une IA")
        else:
            filename = self.SaveEdit.text()
            activate = self.activate.currentText()
            GS = self.GScb.currentText()
            bruit = self.board.bruit
            softgreedy = 0
            if self.board.softgreedy :
                softgreedy = self.board.softgreedy.value()
            np.savez(filename, N=self.board.N, WALLS=self.board.WALLS, W1=self.board.NN[0], W2=self.board.NN[1], activateFunction = activate, GS = GS, bruit = bruit, softgreedy = softgreedy)

    def load(self):
        """
        Load an AI with a given filename
        The file contains the neural network and the game parameters the AI was trained on
        """
        filename = self.LoadEdit.text()
        try :
            data = np.load(filename)
            self.board.N = int(data['N'])
            self.board.WALLS = int(data['WALLS'])
            self.board.NN = (data['W1'], data['W2'])
            self.board.activate = data['activateFunction']
            self.board.strategyGloutonne = np.array_str(data['GS'])
            self.board.bruit = float(data['bruit'])
            self.board.softgreedy = float(data['softgreedy'])
            self.board.G_INIT = self.board.computeGraph()
        except :
            QMessageBox.about(self, "Error", "Path invalide !")

    def launch_train(self):
        """
        Train the AI with the given parameters of the GUI
        """
        if self.board.NN is None:
            QMessageBox.about(self, "Error", "Il faut d'abord créer ou charger une IA")
        else:
            self.updateParameters()
            n_train = int(self.TrainNumber.value())
            self.board.train(self.board.NN, n_train)

    def launch_game(self):

        #get the parameters of the GUI
        self.updateParameters()

        #with AI
        if self.PlayerOne.currentIndex()==0:
            #with AI => compare two AIs
            if self.PlayerTwo.currentIndex()==0:
                if self.board.NN is None:
                    QMessageBox.about(self, "Error", "Il faut d'abord créer ou charger une IA")
                else:
                    self.board.G_INIT = self.board.computeGraph()
                    filename = self.LoadEdit.text()
                    n_compare = self.TrainNumber.value()
                    eps = self.board.EPS
                    self.board.compare(self.board.NN, filename, n_compare, eps)

            # with a Human
            else:
                if self.board.NN is None:
                    QMessageBox.about(self, "Error", "Il faut d'abord créer ou charger une IA")
                else:
                    human = Player_Human('Humain', G=self.board.G_INIT, G_init=self.board.G_INIT, N=self.board.N, WALLS=self.board.WALLS)
                    agent = Player_AI(self.board.NN, 0.0, self.board.activate, None,
                                      'IA', G=self.board.G_INIT, N=self.board.N, WALLS=self.board.WALLS, bruit=self.board.bruit)  # IA joue le mieux possible
                    self.board.play(human, agent)
        else:
            if self.PlayerTwo.currentIndex() == 0:
                if self.board.NN is None:
                    QMessageBox.about(self, "Error", "Il faut d'abord créer ou charger une IA")
                else:
                    human = Player_Human('Humain', G=self.board.G_INIT, G_init=self.board.G_INIT, N=self.board.N, WALLS=self.board.WALLS)
                    agent = Player_AI(self.board.NN, 0.0 ,self.board.activate, None,
                                      'IA', G=self.board.G_INIT, N=self.board.N, WALLS=self.board.WALLS,bruit=self.board.bruit)  # IA joue le mieux possible
                    self.board.play(human, agent)
            else:

                self.board.G_INIT = self.board.computeGraph()
                human1 = Player_Human('Humain 1', G=self.board.G_INIT, G_init=self.board.G_INIT, N=self.board.N, WALLS=self.board.WALLS)
                human2 = Player_Human('Humain 2', G=self.board.G_INIT, G_init=self.board.G_INIT, N=self.board.N, WALLS=self.board.WALLS)
                self.board.play(human1, human2)
                self.board.G_INIT = self.board.computeGraph()

    def create_new(self):
        """
        Create a new AI with the currents parameters of the GUI
        """
        self.updateParameters()
        self.board.NN = createNN(2 * self.board.N ** 2 + 2 * (self.board.N - 1) ** 2 + 2 * (self.board.WALLS + 1), self.NumberNN.value())
        self.board.G_INIT = self.board.computeGraph()

    def updateParameters(self):
        """
        Update the game parameters with the values of the GUI
        """
        self.board.N = self.Size.value()
        self.board.WALLS = self.Wall.value()
        self.board.EPS = self.Epsilon.value()
        self.board.LAMB = self.Lambda.value()
        self.board.LS = self.LScb.currentText()
        self.board.ALPHA = self.Alpha.value()
        self.board.G_INIT = self.board.computeGraph()

        # NOUVELLE METHODES IMPLEMENTEES
        self.board.activate = self.activate.currentText()
        self.board.endEps = self.endEpsilon.value()
        self.board.decayRate = self.decayrateEpsilon.value()
        self.board.strategyGloutonne = self.GScb.currentText()

        # actualise les valeurs du bruit de e-greedy et du % entre soft-max et e-greedy si elles ont été changées
        if self.board.softgreedy != None :
            if type(self.board.softgreedy) != float :
                self.board.softgreedy = self.board.softgreedy.value()

        if self.board.bruit != None :
            if type(self.board.bruit) != float :
                self.board.bruit = self.board.bruit.value()




    # NOUVELLE METHODE IMPLEMENTEE
    def modifyGScb(self):
        """
        Create a new dialog box and asks the how many games we train using Soft-Max and how many games we use E-Greedy
        """
        if self.GScb.currentText() == 'Soft-max et E-greedy' :
            box = QDialog()

            msg = QLabel('Choisissez le % entre E-greedy et soft-max (75% = 75% E-greedy / 25% soft-max)',box)
            msg.move(20,5)

            self.board.softgreedy = QDoubleSpinBox(box)
            self.board.softgreedy.setMinimum(0)
            self.board.softgreedy.setMaximum(100)
            self.board.softgreedy.setSingleStep(1)
            self.board.softgreedy.setValue(50)
            self.board.softgreedy.move(240,30)


            btnBox = QPushButton("Ok", box)
            btnBox.move(260, 80)
            close = lambda: box.close()
            btnBox.clicked.connect(close)


            box.setWindowTitle("Dialog")
            box.setWindowModality(Qt.ApplicationModal)
            box.setGeometry(500, 500, 600, 125)
            box.exec_()

    # NOUVELLE METHODE IMPLEMENTEE
    def modifyBruit(self):
        """
        Create a new dialog box and asks the % of bruit for E-greedy
        """
        if self.GScb.currentText() == 'E-greedy avec bruit':
            box = QDialog()

            msg = QLabel('Choisissez le % de bruit', box)
            msg.move(230, 5)

            self.board.bruit = QDoubleSpinBox(box)
            self.board.bruit.setMinimum(0.1)
            self.board.bruit.setMaximum(1)
            self.board.bruit.setSingleStep(0.01)
            self.board.bruit.setValue(0.1)
            self.board.bruit.move(240, 30)

            btnBox = QPushButton("Ok", box)
            btnBox.move(260, 80)
            close = lambda: box.close()
            btnBox.clicked.connect(close)

            box.setWindowTitle("Dialog")
            box.setWindowModality(Qt.ApplicationModal)
            box.setGeometry(500, 500, 600, 125)
            box.exec_()

    # NOUVELLE METHODE IMPLEMENTEE
    def modifyGScbBruit(self):
        """
        Create a new dialog box and asks the how many games we train using Soft-Max and how many games we use E-Greedy and the % of bruit for E-greedy
        """
        if self.GScb.currentText() == 'Soft-max et E-greedy avec bruit' :
            box = QDialog()

            msg1 = QLabel('Choisissez le % entre E-greedy et soft-max (75% = 75% E-greedy / 25% soft-max)',box)
            msg1.move(20,5)

            self.board.softgreedy = QDoubleSpinBox(box)
            self.board.softgreedy.setMinimum(0)
            self.board.softgreedy.setMaximum(100)
            self.board.softgreedy.setSingleStep(1)
            self.board.softgreedy.setValue(50)
            self.board.softgreedy.move(240,30)

            msg2 = QLabel('Choisissez le % de bruit', box)
            msg2.move(230, 70)

            self.board.bruit = QDoubleSpinBox(box)
            self.board.bruit.setMinimum(0.1)
            self.board.bruit.setMaximum(1)
            self.board.bruit.setSingleStep(0.01)
            self.board.bruit.setValue(0.1)
            self.board.bruit.move(240, 100)

            btnBox = QPushButton("Ok", box)
            btnBox.move(260, 150)
            close = lambda: box.close()
            btnBox.clicked.connect(close)


            box.setWindowTitle("Dialog")
            box.setWindowModality(Qt.ApplicationModal)
            box.setGeometry(500, 500, 600, 200)
            box.exec_()



    def redraw(self,info,msg):
        """
        Redraw the play grid
        :param info: board information in the form of a list obtained with listEncoding function
        :param msg: text to appear beneath the grid, list of two strings
        """
        if info[0] is not None:
            self.draw_map(info)
        self.textGame.setText(msg[0])
        self.textGame.setWordWrap(True)
        self.textTurn.setText(msg[1])
        self.textTurn.setWordWrap(True)
        QApplication.processEvents()


    def reset_all(self,msg):
        self.reset_map()
        self.draw_map()
        self.textGame.setText(msg[0])
        self.textGame.setWordWrap(True)
        self.textTurn.setText(msg[1])
        self.textTurn.setWordWrap(True)




    def draw_map(self,info=[[],[],[],[],[],0,0]):
        """
        Draw the board according to the list encoding received : position player black and white, and the south-west case for the walls
        :param info: list received from ListEnconding function
        """
        self.reset_map()
        pos_playerwhite = [info[0][0],abs(self.board.N - 1 - info[0][1])] if info[0] != [] else info[0]
        pos_player_black = [info[1][0],abs(self.board.N - 1 - info[1][1])] if info[1] != [] else info[1]
        horizontal_walls = self.compute_the_walls(info[2],True)
        vertical_walls = self.compute_the_walls(info[3],False)
        for x in range(0, self.board.N):
            for y in range(0, self.board.N):
                w = Pos(x, y,
                        [y, x] in horizontal_walls,
                        [y, x] in vertical_walls,
                        [y, x] == pos_playerwhite,
                        [y, x] == pos_player_black)
                self.grid_board.addWidget(w, x, y)
        QApplication.processEvents()


    def compute_the_walls(self,walls,horizontal):
        """
        output all the cases touched by the walls (two per wall)
        :param walls: a list of lists
        :return: a list of lists
        """
        res = []
        for wall in walls:
            x, y = wall
            res.append([x , self.board.N - 1 - y])
            if horizontal:
                res.append([x + 1, self.board.N - 1 - y])
            else:
                res.append([x, self.board.N - 2 - y])
        return res

    def reset_map(self):
        """
        Clean the grid to begin from a empty grid
        """
        try:
            while self.grid_board.count():
                item = self.takeAt(0)
                widget = item.widget()

                widget.deleteLater()
        except AttributeError:
            pass



class Pos(QWidget):
    """
    Represents a case of the grid
    """

    def __init__(self, x, y, horizontal= False,vertical=False,
                 playerWhite = False, playerBlack = False):
        super(Pos, self).__init__()

        self.setFixedSize(QSize(60, 60))

        self.x = x
        self.y = y
        self.horizontal = horizontal #part of horizontal wall
        self.vertical = vertical #part of vertical wall
        self.playerWhite = playerWhite #position of white player
        self.playerBlack = playerBlack #position of black player


    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = event.rect()

        outer, inner = Qt.gray, Qt.lightGray

        p.fillRect(r, QBrush(inner))
        pen = QPen(outer)
        pen.setWidth(1)
        p.setPen(pen)
        p.drawRect(r)

        if self.playerWhite:
            p.drawPixmap(r, QPixmap(IMG_WHITE))
        elif self.playerBlack:
            p.drawPixmap(r, QPixmap(IMG_BLACK))
        if self.horizontal:
            pen = QPen(Qt.red, 10)
            p.setPen(pen)
            p.drawLine(0, 0, self.rect().width(), 0)
        if self.vertical:
            pen = QPen(Qt.red, 10)
            p.setPen(pen)
            p.drawLine(self.rect().width(), 0,
                       self.rect().width(), self.rect().height())

        p.end()


class Player_Human():
    def __init__(self, name='Humain', G = None, G_init = None, N = 5, WALLS = 3):
        self.name = name
        self.color = None  # white (0) or black(1)
        self.score = 0
        self.G = G
        self.G_INIT = G_init
        self.N = N
        self.WALLS = WALLS
        self.key = None


    def endGame(self, board, won):
        pass


class Player_AI():
    def __init__(self, NN, eps,activate,learning_strategy, GS , name='IA', G = None, N = 5, WALLS = 3,bruit = 0):
        self.name = name
        self.color = None  # white (0) or black(1)
        self.score = 0
        self.NN = NN
        self.eps = eps
        self.learning_strategy = learning_strategy
        self.G = G
        self.N = N
        self.WALLS = WALLS
        # NOUVELLES METHODES IMPLEMENTEES
        self.activate = activate
        self.GS = GS
        self.bruit = bruit


    def makeMove(self, board, G):
        return self.makeThinkingMove(listMoves(board, self.color, self.N, self.WALLS, G), board, self.color, self.NN, self.eps,self.activate,self.GS,self.learning_strategy)


    def makeThinkingMove(self,moves, s, color, NN, eps,activate, GS ,learning_strategy=None):
        """Fonction appelée pour que l'IA choisisse une action (le nouvel état new_s est retourné)
             à partir d'une liste d'actions possibles "moves" (c'est-à-dire une liste possible d'états accessibles) à partir d'un état actuel (s)
             Un réseau de neurones (NN) est nécessaire pour faire ce choix quand le mouvement n'est pas du hasard.
             Dans le cas greedy (non aléatoire), la couleur du joueur dont c'est le tour sera utilisée pour déterminer s'il faut retenir le meilleur ou le pire coup du point de vue du joueur blanc.
             Le cas greedy survient avec une probabilité 1-eps.
             La stratégie d'apprentissage fait référence à la stratégie utilisée dans la fonction de backpropagation (cfr la fonction de backpropagation pour une description)
             """
        Q_learning = (not learning_strategy is None) and (learning_strategy[0] == 'Q-learning')
        TD_lambda = (not learning_strategy is None) and (learning_strategy[0] == 'TD-lambda')
        # Epsilon greedy
        if GS == 'E-greedy' or GS == 'E-greedy avec bruit':
            greedy = random.random() > eps
            # dans le cas greedy, on recherche le meilleur mouvement (état) possible. Dans le cas du Q-learning (même sans greedy), on a besoin de connaître
            # la probabilité estimée associée au meilleur mouvement (état) possible en vue de réaliser la backpropagation.
            if greedy or Q_learning:
                best_moves = []
                best_value = None
                c = 1
                if color == 1:
                    # au cas où c'est noir qui joue, on s'interessera aux pires coups du point de vue de blanc
                    c = -1
                for m in moves:
                    val = forwardPass(m, NN,activate)
                    if best_value == None or c * val > c * best_value:  # si noir joue, c'est comme si on regarde alors si val < best_value
                        best_moves = [m]
                        n_s = 0
                        # NOUVELLE METHODE IMPLEMENTEE
                        if GS == 'E-greedy avec bruit' :
                            n_s = np.random.normal(0, self.bruit) # bruit e-greedy

                        best_value = val + n_s
                    elif val == best_value:
                        best_moves.append(m)

            if greedy:
                # on prend un mouvement au hasard parmi les meilleurs (pires si noir)
                new_s = best_moves[random.randint(0, len(best_moves) - 1)]

            else:
                # on choisit un mouvement au hasard
                new_s = moves[random.randint(0, len(moves) - 1)]

        # NOUVELLE METHODE IMPLEMENTEE
        elif GS == 'Soft-max' :
            all_moves_prob = [] # liste contenant la distribution de probabilité de chaque mouvement
            all_moves = [] # liste contenant chaque mouvement

            for m in moves :
                val = forwardPass(m,NN,activate)
                all_moves_prob.append(val)
                all_moves.append(m)


            all_moves_prob = softmax(all_moves_prob) # effectue la distribution de probabilité de Soft-Max sur l'array all_moves_prob
            index = random.randint(0, len(all_moves_prob) - 1) # choisi aléatoirement un mouvement dans la liste de mouvements possibles
            p_out_new_s = all_moves_prob[index]
            new_s = all_moves[index]

        # on met à jour les poids si nécessaire
        if Q_learning or TD_lambda:
            p_out_s = forwardPass(s, NN,activate)
            if Q_learning:
                if GS != 'Soft-max' :
                    delta = p_out_s - best_value
                else :
                    delta = p_out_s - p_out_new_s

            elif TD_lambda:
                if GS != 'Soft-max' :
                    if greedy:
                        p_out_new_s = best_value
                    else:
                        p_out_new_s = forwardPass(new_s, NN,activate)

                delta = p_out_s - p_out_new_s
            backpropagation(s, NN, delta,activate,learning_strategy)


        return new_s

    def CalculateEndGame(self,s, won, NN,activate,learning_strategy):
        Q_learning = (not learning_strategy is None) and (learning_strategy[0] == 'Q-learning')
        TD_lambda = (not learning_strategy is None) and (learning_strategy[0] == 'TD-lambda')
        # on met à jour les poids si nécessaire
        if Q_learning or TD_lambda:
            p_out_s = forwardPass(s, NN,activate)
            delta = p_out_s - won
            backpropagation(s, NN, delta,activate,learning_strategy) # do I have to modify also this ?
            if TD_lambda:
                # on remet les eligibility traces à 0 en prévision de la partie suivante
                learning_strategy[3].fill(0)  # remet Z_int à 0
                learning_strategy[4].fill(0)  # remet Z_out à 0

    def endGame(self, board, won):
        self.CalculateEndGame(board, won, self.NN,self.activate,self.learning_strategy)


class Grid_Board(QWidget):
    """
    Game class with all parameters
    """

    #signals for the update of the qwidget
    moved = pyqtSignal(list,list)
    exit = pyqtSignal(list)

    def __init__(self, N=5, Walls=3, eps= 0.3,endEps = 0.1,decayRate = 0.1, alpha = 0.4, lamb = 0.9, ls = 'Q-learning',
                 G = None, G_init = None,NN = None):
        super(Grid_Board,self).__init__()
        self.N = N  # N*N board
        self.WALLS = Walls  # number of walls each player has
        self.EPS = eps  # epsilon in epsilon-greedy
        self.ALPHA = alpha  # learning_rate
        self.LAMB = lamb  # lambda for TD(lambda)
        self.LS = ls  # default learning strategy
        self.G = G  # graph of board (used for connectivity checks)
        self.G_INIT = G_init
        self.NN = NN
        self.current_player = None

        # NOUVELLES METHODES IMPLEMENTEES
        self.activate = None
        self.strategyGloutonne = None
        self.endEps = endEps
        self.decayRate = decayRate
        self.softgreedy = None
        self.bruit = None


    def playGame(self, player1, player2,show=True, delay=0.0):
        # initialization
        players = [player1, player2]
        board = startingBoard(self.N, self.WALLS)
        self.G = self.G_INIT.copy()
        if show:
            self.moved.emit(listEncoding(board, self.N, self.WALLS), ['', MSG_MOVEMENT])
        for i in range(2):
            players[i].color = i
        # main loop
        finished = False
        current_player = 0
        count = 0
        quit = False
        msg = ''
        while not finished:
            if show:
                msg = ''
                txt = ['Blanc', 'Noir ']
                for i in range(2):
                    if i == current_player:
                        msg += '* '
                    else:
                        msg += '  '
                    msg += txt[i] + ' : ' + players[i].name
                    msg += '\n'
                for i in range(2):
                    if 'IA' in players[i].name:
                        # jeu en cours est humain contre IA, on affiche estimation probabilité de victoire pour blanc selon IA
                        p = forwardPass(board, players[i].NN, players[i].activate)
                        msg += '\nEstimation IA : ' + "{0:.4f}".format(p)
                        msg += '\n'
                self.moved.emit(listEncoding(board, self.N, self.WALLS), [msg, MSG_MOVEMENT])
                self.moved.emit(listEncoding(board, self.N, self.WALLS), [msg, MSG_MOVEMENT])
                time.sleep(0.3)
            self.current_player = players[current_player]
            new_board = self.current_player.makeMove(board, self.G) if 'IA' in self.current_player.name else self.makeMove(self.current_player,board)
            # we compute changes of G (if any) to avoid recomputing G at beginning of listMoves
            # we remove the corresponding edges from G
            if not new_board is None:
                v = new_board[2 * self.N ** 2:2 * self.N ** 2 + 2 * (self.N - 1) ** 2] - board[
                                                                                         2 * self.N ** 2:2 * self.N ** 2 + 2 * (
                                                                                                     self.N - 1) ** 2]
                i = v.argmax()
                if v[i] == 1:
                    # a wall has been added, we remove the two corresponding edges of G
                    if i < (self.N - 1) ** 2:
                        # horizontal wall
                        a, b = i % (self.N - 1), i // (self.N - 1)
                        E = [[a, b, 1], [a, b + 1, 3], [a + 1, b, 1], [a + 1, b + 1, 3]]
                    else:
                        # vertical wall
                        a, b = (i - (self.N - 1) ** 2) % (self.N - 1), (i - (self.N - 1) ** 2) // (self.N - 1)
                        E = [[a, b, 0], [a + 1, b, 2], [a, b + 1, 0], [a + 1, b + 1, 2]]
                    for e in E:
                        self.G[e[0]][e[1]][e[2]] = 0
            board = new_board

            if show :
                self.moved.emit(listEncoding(board,self.N,self.WALLS),[msg,MSG_MOVEMENT]) if board is not None else None

            if board is None:
                # human player quit
                quit = True
                finished = True
                if show :
                    self.exit.emit(['',END_GAME])

            elif endOfGame(board, self.N):
                players[current_player].score += 1
                white_won = current_player == 0
                players[current_player].endGame(board, white_won)
                if show:
                    self.moved.emit(listEncoding(board, self.N, self.WALLS),['',END_GAME])
                    time.sleep(0.1)
                finished = True
            else:
                current_player = (current_player + 1) % 2
        return quit

    def computeGraph(self, board=None):
        # order of steps in edge encoding: (1,0), (0,1), (-1,0), (0,-1)
        pos_steps = [(1, 0), (0, 1)]
        for i in range(len(pos_steps)):
            pos_steps[i] = np.array(pos_steps[i])
        g = np.zeros((self.N, self.N, 4))
        for i in range(self.N):
            for j in range(self.N):
                c = np.array([i, j])
                for k in range(2):
                    s = pos_steps[k]
                    if board is None:
                        # initial setup
                        new_c = c + s
                        if new_c.min() >= 0 and new_c.max() <= self.N - 1:
                            g[i][j][k] = 1
                            g[new_c[0]][new_c[1]][k + 2] = 1
                    else:
                        if canMove(board, c, s, self.N, self.WALLS):
                            new_c = c + s
                            g[i][j][k] = 1
                            g[new_c[0]][new_c[1]][k + 2] = 1
        return g

    def train(self, NN, n_train=10000):
        """
        Train the IA with current parameters
        :param NN: neural network created according to the parameters of the GUI
        :param n_train: number of turns to train the IA
        """

        if self.LS == 'Q-learning':
            learning_strategy1 = (self.LS, self.ALPHA)
            learning_strategy2 = (self.LS, self.ALPHA)
        elif self.LS == 'TD-lambda':
            learning_strategy1 = (self.LS, self.ALPHA, self.LAMB, np.zeros(NN[0].shape), np.zeros(NN[1].shape))
            learning_strategy2 = (self.LS, self.ALPHA, self.LAMB, np.zeros(NN[0].shape), np.zeros(NN[1].shape))

        # NOUVELLES METHODES IMPLEMENTEES
        eps_copy = self.EPS # copie la valeur de EPS pour ensuite la modifier si elle doit être décroissante
        end_eps = self.endEps
        decay_rate = self.decayRate
        strategyGloutonne = self.strategyGloutonne
        # training session
        for j in range(n_train):
            if 'Soft-max et E-greedy' in self.strategyGloutonne :
                if j < ( (self.softgreedy / 100) * n_train):
                    strategyGloutonne = 'E-greedy avec bruit' if 'E-greedy avec bruit' in self.strategyGloutonne else 'E-greedy'
                else :
                    strategyGloutonne = 'Soft-max'
            self.moved.emit([None],['\nEntraînement (' + str(n_train) + ' parties)',progressBar(j, n_train)])

            if self.bruit != 0 :
                agent1 = Player_AI(NN, eps_copy,self.activate,learning_strategy1,strategyGloutonne, 'IA 1', G=self.G_INIT, N=self.N, WALLS=self.WALLS,bruit=self.bruit)
                agent2 = Player_AI(NN, eps_copy,self.activate,learning_strategy2,strategyGloutonne, 'IA 2', G=self.G_INIT, N=self.N, WALLS=self.WALLS,bruit=self.bruit)
            else :
                agent1 = Player_AI(NN, eps_copy, self.activate, learning_strategy1, strategyGloutonne, 'IA 1',G=self.G_INIT, N=self.N, WALLS=self.WALLS)
                agent2 = Player_AI(NN, eps_copy, self.activate, learning_strategy2, strategyGloutonne, 'IA 2',G=self.G_INIT, N=self.N, WALLS=self.WALLS)
            # NOUVELLES METHODES IMPLEMENTEES
            if (eps_copy - decay_rate) > end_eps: # fait décroire EPS tant qu'on a pas atteint le EPS final
                eps_copy = eps_copy - decay_rate
            self.playGame(agent1, agent2,show=False)

        self.moved.emit([None], ['\nEntraînement (' + str(n_train) + ' parties)', 'Entraînement terminé !'])


    def compare(self, NN1, filename, n_compare=1000, eps=0.05):
        """
        Compare 2 IA and prints the results of IA 1
        :param NN1: Neural network of AI1
        :param filename: file containing data for the AI 2
        :param n_compare: Number of comparisons
        :param eps: epsilon parameter
        """

        data = np.load(filename)
        NN2 = (data['W1'], data['W2'])
        # NOUVELLE METHODE IMPLEMENTEE
        activate2 = data['activateFunction']
        GS2 = np.array_str(data['GS'])
        bruit2 = data['bruit']
        softgreedy2 = data['softgreedy']


        msg = '\nTournoi IA vs ' + filename + ' (' + str(n_compare) + ' parties, eps=' + str(eps) + ')'

        i = 0

        strategyGloutonne = self.strategyGloutonne
        strategyGloutonne2 = GS2

        for j in range(n_compare):
            # NOUVELLES METHODES IMPLEMENTEES
            if 'Soft-max et E-greedy' in self.strategyGloutonne :
                if j < ( (self.softgreedy / 100) * n_compare):
                    strategyGloutonne = 'E-greedy avec bruit' if 'E-greedy avec bruit' in self.strategyGloutonne else 'E-greedy'
                else :
                    strategyGloutonne = 'Soft-max'

            if 'Soft-max et E-greedy' in GS2 :
                if j < ( (softgreedy2 / 100) * n_compare):
                    strategyGloutonne2 = 'E-greedy avec bruit' if 'E-greedy avec bruit' in GS2 else 'E-greedy'
                else :
                    strategyGloutonne2 = 'Soft-max'

            agent1 = Player_AI(NN1, eps, self.activate, None, strategyGloutonne , 'IA 1', G=self.G_INIT, N=self.N,WALLS=self.WALLS, bruit=self.bruit)
            agent2 = Player_AI(NN2, eps, activate2, None, strategyGloutonne2 , 'IA 2', G=self.G_INIT, N=self.N, WALLS=self.WALLS,bruit=bruit2)
            players = [agent1, agent2]
            self.moved.emit([None], [msg, progressBar(j, n_compare)])
            self.playGame(players[i], players[(i + 1) % 2], show=False)
            i = (i + 1) % 2
        perf = agent1.score / n_compare
        msg_end = "Parties gagnées par l'IA : {0:.2f} %".format(perf * 100)
        self.moved.emit([None], [msg, msg_end])
        #waitForKey()


    def play(self, player1, player2, delay=0.1):
        """
        Play a game with at least one human
        """
        i = 0
        players = [player1, player2]
        quit = False
        while not quit:
            quit = self.playGame(players[i], players[(i + 1) % 2], True, delay)
            i = (i + 1) % 2



    def makeMove(self,player, board):
        """
        Make a move for a human player, receive input through keys
        :param player: information about the player (class Player_Human)
        :param board: current info about the positions of players and walls
        """
        moves = listMoves(board, player.color, self.N, self.WALLS, self.G)
        lmoves = [listEncoding(m ,self.N, self.WALLS) for m in moves]
        lboard = listEncoding(board,self.N,self.WALLS)
        D = [[LEFT, [-1, 0], None], [RIGHT, [1, 0], None], [UP, [0, 1], None], [DOWN, [0, -1], None],  # one step moves
             [LEFT, [-2, 0], None], [RIGHT, [2, 0], None], [UP, [0, 2], None], [DOWN, [0, -2], None],  # jumps
             [UP_LEFT, [-1, 1], None], [UP_RIGHT, [1, 1], None], [DOWN_LEFT, [-1, -1], None],
             [DOWN_RIGHT, [1, -1], None]]  # diagonal moves
        for i in range(len(D)):
            for j in range(len(lmoves)):
                m = moves[j]
                lm = lmoves[j]
                if list(np.array(lm[player.color])) == list(np.array(lboard[player.color]) + np.array(D[i][1])):
                    D[i][2] = m
                    break
        wall_moves = [lm for lm in lmoves if lm[player.color] == lboard[player.color]]
        wall_coord = [[], []]
        wall_hv = [[], []]
        for lm in wall_moves:
            i = int(lm[2] == lboard[2])
            wall_hv[i].append(lm)
            for c in lm[2 + i]:
                if c not in lboard[2 + i]:
                    break
            wall_coord[i].append(c)
        quit = False
        key = waitForKey()
        while not quit:
            if key == QUIT or key == "q":
                quit = True
                break
            # player changes position:
            for i in range(len(D)):
                if key == D[i][0]:
                    if not (D[i][2] is None):
                        return D[i][2]
                    elif (i <= 3) and (D[i + 4][2] is None):
                        p = np.array(lboard[player.color]) + np.array(D[i][1])
                        q = np.array(lboard[(player.color + 1) % 2])
                        if p.tolist() == q.tolist():
                            # we moved to the opponent's position but the jump is blocked, we check if some diagonal move is possible
                            s = np.array(D[i][1])
                            diagonal_jump_feasible = False
                            for j in range(8, 12):
                                if not (D[j][2] is None):
                                    r = np.array(D[j][1])
                                    if r[0] == s[0] or r[1] == s[1]:
                                        diagonal_jump_feasible = True
                            if diagonal_jump_feasible:
                                halfway = board.copy()
                                halfway[player.color * self.N ** 2:self.color * self.N ** 2 + self.N ** 2] = halfway[((player.color + 1) % 2) * self.N ** 2 + self.N ** 2]
                                self.moved.emit(listEncoding(halfway,self.N, self.WALLS),['',MSG_DIAGONAL])
                                second_key = waitForKey()
                                diagonal_jump = False
                                for j in range(4):
                                    if second_key == D[j][0]:
                                        t = np.array(D[j][1])
                                        r = s + t
                                        if abs(r[0]) == 1 and abs(r[1]) == 1:
                                            # diagonal jump selected, we check if that jump is feasible
                                            for k in range(8, 12):
                                                if r.tolist() == D[k][1] and not (D[k][2] is None):
                                                    key = D[k][0]
                                                    diagonal_jump = True
                                                    break
                                if not diagonal_jump:
                                    self.moved.emit(listEncoding(board,self.N,self.WALLS), ['',MSG_MOVEMENT])
                        else:
                            self.moved.emit(listEncoding(board, self.N, self.WALLS), ['Mouvement invalide', MSG_MOVEMENT])
                            self.moved.emit(listEncoding(board, self.N, self.WALLS),
                                            ['Mouvement invalide', MSG_MOVEMENT])
                            key = waitForKey()

            # player puts down a wall
            if key == "w" and lboard[4 + player.color] > 0 and len(wall_moves) > 0:
                msg = "Pose d'un mur (flèches pour faire defiler, w pour changer l'orientation, ENTER pour selectionner, a pour annuler, q pour quitter)"
                j = 0
                if len(wall_hv[0]) > 0:
                    h = 0
                else:
                    h = 1
                while not quit:
                    i = lmoves.index(wall_hv[h][j])
                    self.moved.emit(listEncoding(moves[i], self.N, self.WALLS),['',msg])
                    self.moved.emit(listEncoding(moves[i], self.N, self.WALLS), ['', msg])
                    key = waitForKey()
                    if key == QUIT or key == 'q':
                        quit = True
                        break
                    if key == 'a':
                        self.moved.emit(listEncoding(board, self.N, self.WALLS),['',MSG_MOVEMENT])
                        break
                    elif key == 'w':
                       if len(wall_hv[(h + 1) % 2]) > 0:
                            c = wall_coord[h][j]
                            h = (h + 1) % 2
                            if c in wall_coord[h]:
                                j = wall_coord[h].index(c)
                            else:
                                best_d = wall_coord[h][0]
                                min_val = (abs(best_d[0] - c[0]) + abs(best_d[1] - c[1])) ** 2
                                for d in wall_coord[h]:
                                    val = (abs(d[0] - c[0]) + abs(d[1] - c[1])) ** 2
                                    if val < min_val:
                                        min_val = val
                                        best_d = d
                                j = wall_coord[h].index(best_d)
                    elif key == LEFT:
                        j = (j - 1) % len(wall_hv[h])
                    elif key == RIGHT:
                        j = (j + 1) % len(wall_hv[h])
                    elif key == UP:
                        c = wall_coord[h][j]
                        next_j = j
                        for k in range(j, len(wall_coord[h])):
                            if wall_coord[h][k][0] == c[0] and wall_coord[h][k][1] > c[1]:
                                next_j = k
                                break
                        j = next_j
                    elif key == DOWN:
                        c = wall_coord[h][j]
                        next_j = j
                        for k in range(j, -1, -1):
                            if wall_coord[h][k][0] == c[0] and wall_coord[h][k][1] < c[1]:
                                next_j = k
                                break
                        j = next_j
                    elif key == '\r':
                        return moves[i]
        if quit:
            return None

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
