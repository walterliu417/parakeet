import helperfuncs
from helperfuncs import *
from typing import Optional

class Node:
    pass
class Node:

    def __init__(self, board: chess.Board, net: onnxruntime.InferenceSession, move: Optional[chess.Move]=None, parent: Optional[Node]=None, depth=0):
        self.board = board
        self.move = move
        self.value = None
        self.parent = parent
        self.visits = 0
        self.depth = depth

        self.net = net
        self.children = []
        self.flag = None

        try:
            self.capture = self.parent.board.is_capture(self.move)
        except:
            self.capture = False

        try:
            self.check = self.parent.board.is_check(self.move)
        except:
            self.check = False

        try:
            if self.move.promotion is not None:
                self.promotion = True
            else:
                self.promotion = False
        except:
            self.promotion = False


    def ucb(self, time_fraction):
        bonus = 0
        if self.capture:
            bonus = quiescent
        elif self.check:
            bonus = quiescent / 2
        elif self.promotion:
            bonus = quiescent
        return self.value - factor * (1 - decay * time_fraction) * np.sqrt(np.log(self.parent.visits + 1) / (self.visits + 1)) - (bonus * factor * (1 - decay * time_fraction))

    def evaluate_nn(self):
        boardlist = fast_board_to_boardmap(self.board)
        if not self.board.turn:
            boardlist = np.rot90(boardlist, 2) * -1
            boardlist = boardlist.tolist()
        pos = np.array(boardlist).astype(np.float32).reshape(1, 1, 8, 8)
        ort_inputs = {"input": pos}
        return self.net.run(None, ort_inputs)[0]

    def evaluate_position(self):
        if TABLEBASE and lt5(self.board):
            result = TABLEBASE.probe_wdl(self.board)
            if result == 2:
                return 1
            elif result == -2:
                return 0
            elif result in [-1, 0, 1]:
                return 0.5
        outcome = self.board.result(claim_draw=True)
        if outcome != "*":
            if (outcome == "1-0") and self.board.turn:
                return 1 + max((10 - self.depth) / 10, 0)
            elif (outcome == "0-1") and self.board.turn:
                return 0 - max((10 - self.depth) / 10, 0)
            elif (outcome == "1-0") and not self.board.turn:
                return 0 - max((10 - self.depth) / 10, 0)
            elif (outcome == "0-1") and not self.board.turn:
                return 1 + max((10 - self.depth) / 10, 0)
            elif outcome == "1/2-1/2":
                return 0.5
        return None

    def generate_children(self):
        all_positions = []

        evaled = []
        not_evaled = []
        blm = self.board.legal_moves
        helperfuncs.nodes += blm.count()
        for move in blm:
            newboard = self.board.copy()
            newboard.push(move)
            newnode = Node(newboard, self.net, move, self, depth=self.depth + 1)
            if newnode.board.halfmove_clock > 50:
                newnode.value = 0.5
                evaled.append(newnode)
            else:
                score = newnode.evaluate_position()
                if score is not None:
                    newnode.value = score
                    evaled.append(newnode)
                else:
                    boardlist = fast_board_to_boardmap(newboard)
                    if not newboard.turn:
                        boardlist = np.rot90(boardlist, 2) * -1
                        boardlist = boardlist.tolist()
                    all_positions.append(boardlist)
                    not_evaled.append(newnode)

        pos = np.array(all_positions).astype(np.float32).reshape(len(not_evaled), 1, 8, 8)
        ort_inputs = {"input": pos}
        while True:
            try:
                # Weird BatchNorm error that happens sometimes?? Attempt to recover.
                result = self.net.run(None, ort_inputs)[0]
                break
            except:
                continue
        for i in range(len(not_evaled)):
            not_evaled[i].value = float(result[i])
            evaled.append(not_evaled[i])

        self.children = evaled


    def pns(self, start_time, time_for_this_move) -> Node:
        while time.time() - start_time < time_for_this_move:
            # 1. Traverse tree with UCT + quiescence and decreasing exploration with time.
            target_node = self
            while target_node.children != []:
                target_node.visits += 1
                time_fraction = (time.time() - start_time) / time_for_this_move
                target_node = min(target_node.children, key=lambda child: child.ucb(time_fraction))

            # 2. Expansion and simulation
            target_node.generate_children()
            target_node.visits += 1

            # 3. Backpropagation
            while True:
                if target_node.children == []:
                    target_node.value = target_node.evaluate_position()
                    if target_node.value is None:
                        target_node.value = target_node.evaluate_nn()
                else:
                    target_node.value = 1 - min(target_node.children, key=lambda child: child.value).value
                if target_node.parent is not None:
                    target_node = target_node.parent
                else:
                    break

        # 4. Select move - UBFMS
        max_visits = max(self.children, key=lambda child: child.visits)
        selected_child = min(self.children, key=lambda child: child.value)
        print(f"info string root_visits {self.visits} max_visits {max_visits.visits} best_visits {selected_child.visits}")
        return selected_child
