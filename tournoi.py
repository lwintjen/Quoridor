"""
Auteur : Loris Wintjens
Titre : Tournoi - INFO-F106
But : Implémentation d'une IA optimisée pour un tournoi
Date : 7/04/19
"""
import sys
from IA_partie2 import *
from utils_partie2 import *
from partie4 import Player_AI, Player_Human


def computeGraph(N, WALLS, board=None):
    # order of steps in edge encoding: (1,0), (0,1), (-1,0), (0,-1)
    pos_steps = [(1, 0), (0, 1)]
    for i in range(len(pos_steps)):
        pos_steps[i] = np.array(pos_steps[i])
    g = np.zeros((N, N, 4))
    for i in range(N):
        for j in range(N):
            c = np.array([i, j])
            for k in range(2):
                s = pos_steps[k]
                if board is None:
                    # initial setup
                    new_c = c + s
                    if new_c.min() >= 0 and new_c.max() <= N - 1:
                        g[i][j][k] = 1
                        g[new_c[0]][new_c[1]][k + 2] = 1
                else:
                    if canMove(board, c, s, N, WALLS):
                        new_c = c + s
                        g[i][j][k] = 1
                        g[new_c[0]][new_c[1]][k + 2] = 1
    return g


def train(NN,N,WALLS,G_INIT, n_train=10000):
    """
    Train the IA with current parameters
    :param NN: neural network created according to the parameters of the GUI
    :param n_train: number of turns to train the IA
    """

    learning_strategy1 = ('Q-Learning', 0.4)
    learning_strategy2 = ('Q-Learning', 0.4)

    # NOUVELLES METHODES IMPLEMENTEES
    strategyGloutonne = 'Soft-max'
    # training session
    for j in range(n_train):

        agent1 = Player_AI(NN, 0, 'Leaky ReLU', learning_strategy1, strategyGloutonne, 'IA 1',
                           G=G_INIT, N=N, WALLS=WALLS)
        agent2 = Player_AI(NN, 0, 'Leaky ReLU', learning_strategy2, strategyGloutonne, 'IA 2',
                           G=G_INIT, N=N, WALLS=WALLS)
        playGame(agent1, agent2, show=False)

def progressBar(i, n):
    return '  ' + str(int(100 * i / n)) + '%'

def playGame(player1, player2, show = False, delay = 0.0):
    global N, WALLS, G, G_INIT
    # initialization
    players = [ player1, player2 ]
    board = startingBoard(N,WALLS)
    G = G_INIT.copy()
    for i in range(2):
        players[i].color = i
    # main loop
    finished = False
    current_player = 0
    count = 0
    quit = False
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
                msg+='\n'
            for i in range(2):
                if players[i].name=='IA':
                    # jeu en cours est humain contre IA, on affiche estimation probabilité de victoire pour blanc selon IA
                    p = forwardPass(board, players[i].NN)
                    msg+='\nEstimation IA : ' + "{0:.4f}".format(p)
                    msg+='\n'
            display(board, msg)
            time.sleep(delay)
        new_board = players[current_player].makeMove(board, G)
        # we compute changes of G (if any) to avoid recomputing G at beginning of listMoves
        # we remove the corresponding edges from G
        if not new_board is None:
            v = new_board[2*N**2:2*N**2 + 2*(N-1)**2] - board[2*N**2:2*N**2 + 2*(N-1)**2]
            i = v.argmax()
            if v[i] == 1:
                # a wall has been added, we remove the two corresponding edges of G
                if i < (N-1)**2:
                    # horizontal wall
                    a, b = i%(N-1), i//(N-1)
                    E = [ [a,b,1], [a,b+1,3], [a+1,b,1], [a+1,b+1,3] ]
                else:
                    # vertical wall
                    a, b = (i - (N-1)**2)%(N-1), (i - (N-1)**2)//(N-1)
                    E = [ [a,b,0], [a+1,b,2], [a,b+1,0], [a+1,b+1,2] ]
                for e in E:
                    G[e[0]][e[1]][e[2]] = 0
        board = new_board
        if board is None:
            # human player quit
            quit = True
            finished = True
        elif endOfGame(board,N):
            players[current_player].score += 1
            white_won = current_player == 0
            players[current_player].endGame(board, white_won)
            if show:
                display(board)
                time.sleep(0.3)
            finished = True
        else:
            current_player = (current_player+1)%2
    return quit



if __name__ == '__main__':
    data = sys.argv
    # fonction d'activation = Leaky ReLU, méthode gloutonne = Soft-Max, learning strategy = Q-Learning
    # nb de neurones = 40, learning rate = 0.40
    print('Leaky ReLU')
    n_train = int(data[1])
    filename = data[2]
    N = 5
    WALLS = 3
    NN = createNN(2 * N ** 2 + 2 * (N - 1) ** 2 + 2 * (WALLS + 1), 40)
    G_INIT = computeGraph(N,WALLS)
    train(NN,N,WALLS,G_INIT,n_train)

    np.savez(filename, N=N, WALLS=WALLS, W1=NN[0], W2=NN[1])