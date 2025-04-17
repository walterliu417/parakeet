import numpy as np
import chess
import helperfuncs
from helperfuncs import *
from search import *

class Parrot:

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.time_remaining = 0
        self.root_node = None  

        sess_options = onnxruntime.SessionOptions()

        self.model = onnxruntime.InferenceSession(model_path, sess_options, providers=[provider])
        print("Current best model loaded successfully!")

    def set_fen(self, fen):
        self.board = chess.Board(fen)

    def search(self, movetime, wtime, btime):
        helperfuncs.nodes = 0
        if not self.root_node:
            self.root_node = Node(self.board, self.model, None, None)
        else:
            tt = False
            for child in self.root_node.children:
                if child.board.fen() == self.board.fen():
                    tt = True
                    self.root_node = child
                    break
            if not tt: self.root_node = Node(self.board, self.model, None, None)
        self.root_node.parent = None # Clear memory used by previous node
        
        if movetime > 0:
            self.time_for_this_move = movetime
        else:
            # Simple time management
            if self.board.turn == chess.WHITE:
                self.time_remaining = wtime
            elif self.board.turn == chess.BLACK:
                self.time_remaining = btime
                
            if self.board.fullmove_number < 5:
                # Opening - save time
                self.time_for_this_move = self.time_remaining * 0.4 / 25
            elif self.board.fullmove_number < 30:
                # Midgame - use time
                self.time_for_this_move = self.time_remaining * 0.8 / 25
            else:
                # Endgame
                self.time_for_this_move = (self.time_remaining / 25)
<<<<<<< HEAD
=======

>>>>>>> 1b1e1d15a30440e15800f5c3bd4a10a79ab4b4c3
        child = self.root_node.pns(time.time(), self.time_for_this_move)

        movelist = []
        bestmove = child.move
        cp = nn_to_cp(self.root_node.value)
        while child.children != []:
            movelist.append(child.move.uci())
            child = min(child.children, key=lambda c: c.value)
        nps = helperfuncs.nodes / self.time_for_this_move
        print(f"info depth 1 seldepth {len(movelist)} time {self.time_for_this_move * 1000} nodes {helperfuncs.nodes} score {cp} nps {nps} pv {' '.join(movelist)}")
        return bestmove


def run():
    while True:
        command = input()
        if command == "":
            continue
        command = command.split()        
        if command[0] == "uci":
            print("id name Parrot v0.0")
            print("id author Walter Liu")
            print("option name explore_factor type spin default 100 min 0 max 100")
            print("option name quiescent_bonus type spin default 50 min 0 max 500")
            print("option name explore_decay type spin default 100 min 0 max 100")
            print("option name tablebase_dir type string default /content/drive/MyDrive/parrot/tablebase_5pc")
            print("option name net_path type string default parrot.pickle")
            print("uciok")
        elif command[0] == "isready":
            print("readyok")
        elif command[0] == "ucinewgame":
            engine = Parrot()
        elif command[0] == "quit":
            import sys
            sys.exit()
        elif command[0] == "position":
            if command[1] == "fen":
                engine.set_fen(' '.join(command[2:]))
<<<<<<< HEAD
                if len(command) > 3:
                    if command[3] == "moves":
                        for move in command[4:]:
                            engine.board.push_uci(move)
=======
>>>>>>> 1b1e1d15a30440e15800f5c3bd4a10a79ab4b4c3
            elif command[1] == "startpos":
                engine.set_fen(chess.STARTING_FEN)
                if len(command) > 2:
                    if command[2] == "moves":
                        engine.set_fen(chess.STARTING_FEN)
                        for move in command[3:]:
                            engine.board.push_uci(move)
        elif command[0] == "go":  # TODO: command parsing needs to be redone
            movetime = 0
            wtime = 0
            btime = 0
<<<<<<< HEAD
            if command[1] == "movetime":
                movetime = float(command[2]) / 1000.0
            if command[1] == "wtime":
                wtime = float(command[2]) / 1000.0
            if command[1] == "btime":
                btime = float(command[2]) / 1000.0
            if len(command) > 3 and command[3] == "btime":
                btime = float(command[4]) / 1000.0
=======
            if len(command) > 1:
                if command[1] == "movetime":
                    movetime = int(command[2]) / 1000
            if len(command) > 3:
                if command[1] == "wtime":
                    wtime = int(command[2]) / 1000
                if command[3] == "btime":
                    btime = int(command[4]) / 1000
>>>>>>> 1b1e1d15a30440e15800f5c3bd4a10a79ab4b4c3
            print(f"bestmove {engine.search(movetime, wtime, btime)}")
        elif command[0] == "setoption":
            name = command[2]
            if name == "explore_factor":
<<<<<<< HEAD
                helperfuncs.factor = float(command[4]) / 100.0
            elif name == "quiescent_bonus":
                helperfuncs.quiescent = float(command[4]) / 1000.0
            elif name == "explore_decay":
                helperfuncs.decay = float(command[4]) / 100.0
=======
                helperfuncs.factor = float(command[4]) / 100
            elif name == "quiescent_bonus":
                helperfuncs.quiescent = float(command[4]) / 1000
            elif name == "explore_decay":
                helperfuncs.decay = float(command[4]) / 100
>>>>>>> 1b1e1d15a30440e15800f5c3bd4a10a79ab4b4c3
            elif name == "tablebase_dir":
                try:
                    TABLEBASE = chess.syzygy.open_tablebase(command[4])
                    print(f"Tablebase found at {command[4]}.")
                except:
                    print("Tablebase not found!")
            elif name == "net_path":
                model_path = command[4]

if __name__ == "__main__":
    run()
