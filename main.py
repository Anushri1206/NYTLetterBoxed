import argparse
from typing import List, Set, Union
from collections import defaultdict, deque
import heapq
import time
import math  # Import math
from itertools import product  # Import product


class WordTrieNode:
    def __init__(self, value: str, parent: Union['WordTrieNode', None]):
        self.value = value
        self.parent = parent
        self.children = {}
        self.valid = False

    def get_word(self) -> str:
        if self.parent is not None:
            return self.parent.get_word() + self.value
        else:
            return self.value


class LetterBoxed:
    def __init__(self, input_string: str, dictionary: str):
        self.input_string = input_string.lower()
        self.sides = {side for side in input_string.split('-')}
        self.puzzle_letters = {letter for side in self.sides for letter in side}

        # Create a mapping from letter to bitmask index
        self.letter_to_bitmask = {letter: 1 << i for i, letter in enumerate(sorted(self.puzzle_letters))}
        self.all_letters_mask = (1 << 12) - 1

        self.root = WordTrieNode('', None)
        with open(dictionary) as f:
            for line in f.readlines():
                self.add_word(line.strip().lower())

        self.puzzle_words = self.get_puzzle_words()
        self.puzzle_words.sort(key=len, reverse=True)  # Sort by length (descending)

        self.puzzle_graph = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for word in self.puzzle_words:
            # Use bitmasks in puzzle_graph
            letter_mask = 0
            for char in word:
                letter_mask |= self.letter_to_bitmask[char]
            self.puzzle_graph[word[0]][word[-1]][letter_mask].append(word)

    def add_word(self, word) -> None:
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = WordTrieNode(char, node)
            node = node.children[char]
        node.valid = True

    def _puzzle_words_inner(self, node: WordTrieNode, last_side: str) -> List[WordTrieNode]:
        valid_nodes = [node] if node.valid else []
        if node.children:
            for next_side in self.sides - {last_side}:
                for next_letter in next_side:
                    if next_letter in node.children:
                        next_node = node.children[next_letter]
                        valid_nodes += self._puzzle_words_inner(next_node, next_side)
        return valid_nodes

    def get_puzzle_words(self) -> List[str]:
        all_valid_nodes = []
        for starting_side in self.sides:
            for starting_letter in starting_side:
                if starting_letter in self.root.children:
                    all_valid_nodes += self._puzzle_words_inner(self.root.children[starting_letter], starting_side)
        initial_words = [node.get_word() for node in all_valid_nodes]

        # Pre-filter puzzle_words
        useful_words = []
        for word in initial_words:
            word_mask = 0
            for char in word:
                word_mask |= self.letter_to_bitmask[char]
            is_useful = False
            for other_word in initial_words:
                if word == other_word:
                    continue
                other_word_mask = 0
                for char in other_word:
                    other_word_mask |= self.letter_to_bitmask[char]
                if (word_mask & ~other_word_mask) != 0:
                    is_useful = True
                    break
            if is_useful:
                useful_words.append(word)

        return useful_words

    def _find_solutions_inner(self, path_words: List[List[str]], letters_mask: int, next_letter: str, max_len: int) -> List[List[List[str]]]:
        if letters_mask == self.all_letters_mask:
            return [path_words]
        if len(path_words) >= max_len:
            return []
        
        solutions = []
        next_letter_edges = self.puzzle_graph[next_letter]

        for last_letter in next_letter_edges:
            for letter_edge_mask, edge_words in next_letter_edges[last_letter].items():
                if letter_edge_mask & ~letters_mask:
                    new_solutions = self._find_solutions_inner(
                        path_words + [edge_words], letters_mask | letter_edge_mask, last_letter, max_len
                    )
                    solutions.extend(new_solutions)
        return solutions
    
    def find_best_solution_dfs(self) -> List[str]:
        start_time = time.time()
        best_solution = []
        for max_len in range(1, 6):
            for first_letter in self.puzzle_letters:
                for last_letter in self.puzzle_letters:
                    for letter_edge_mask, edge_words in self.puzzle_graph[first_letter][last_letter].items():
                        result = self._find_solutions_inner([edge_words], letter_edge_mask, last_letter, max_len)
                        if result:
                            for res in result:
                                if not best_solution or len(res) < len(best_solution):
                                    best_solution = res
            if best_solution:
                flat_solution = [word for word_list in best_solution for word in word_list]
                end_time = time.time()
                print(f"DFS Time: {end_time - start_time:.4f} seconds")
                return flat_solution

        end_time = time.time()
        print(f"DFS Time: {end_time - start_time:.4f} seconds (No solution found)")
        return []
    
    def heuristic(self, letters_mask):
        remaining_letters = bin(self.all_letters_mask & ~letters_mask).count('1')
        if remaining_letters == 0:
            return 0

        if not hasattr(self, 'max_letters_data'):
            self.max_letters_data = {}

        if letters_mask not in self.max_letters_data:
            max_letters_per_word = 0
            for word in self.puzzle_words:
                word_mask = 0
                for char in word:
                    word_mask |= self.letter_to_bitmask.get(char, 0)
                covered_count = bin(word_mask & (~letters_mask) & self.all_letters_mask).count('1')
                max_letters_per_word = max(max_letters_per_word, covered_count)
            self.max_letters_data[letters_mask] = max_letters_per_word

        max_letters_per_word = self.max_letters_data[letters_mask]

        if max_letters_per_word == 0:
            return float('inf')  # No word covers any new letters

        return math.ceil(remaining_letters / max_letters_per_word)  # Use ceil for admissibility


    def find_best_solution_astar(self) -> List[str]:
        start_time = time.time()
        best_solution_length = float('inf')  # Initialize with infinity
        open_set = []

        for first_letter, last_letter in product(self.puzzle_letters, repeat=2):
            for letter_edge_mask, edge_words in self.puzzle_graph[first_letter][last_letter].items():
                initial_state = ([edge_words], letter_edge_mask, last_letter)
                g_score = 1
                h_score = self.heuristic(letter_edge_mask)
                f_score = g_score + h_score
                heapq.heappush(open_set, (f_score, -g_score, initial_state))

        visited = set()

        while open_set:
            f_score, neg_g_score, (path_words, letters_mask, next_letter) = heapq.heappop(open_set)
            g_score = -neg_g_score

            # Early pruning based on best solution length
            if g_score + self.heuristic(letters_mask) >= best_solution_length:
                continue

            state_tuple = (letters_mask, next_letter, len(path_words))
            if state_tuple in visited:
                continue
            visited.add(state_tuple)

            if letters_mask == self.all_letters_mask:
                flat_solution = [word for word_list in path_words for word in word_list]
                best_solution_length = len(flat_solution)
                end_time = time.time()
                print(f"A* Time: {end_time - start_time:.4f} seconds")
                return flat_solution

            if len(path_words) >= 5:
                continue

            next_letter_edges = self.puzzle_graph[next_letter]
            for last_letter in next_letter_edges:
                for letter_edge_mask, edge_words in next_letter_edges[last_letter].items():
                    if letter_edge_mask & ~letters_mask:
                        new_letters_mask = letters_mask | letter_edge_mask
                        new_path_words = path_words + [edge_words]
                        new_g_score = len(new_path_words)
                        new_h_score = self.heuristic(new_letters_mask)
                        new_f_score = new_g_score + new_h_score
                        new_state = (new_path_words, new_letters_mask, last_letter)
                        new_state_tuple = (new_letters_mask, last_letter, len(new_path_words))

                        if new_state_tuple not in visited:
                            heapq.heappush(open_set, (new_f_score, -new_g_score, new_state))


        end_time = time.time()
        print(f"A* Time: {end_time - start_time:.4f} seconds (No solution found)")
        return []

    def find_best_solution(self, use_astar=False):
      if use_astar:
        return self.find_best_solution_astar()
      else:
          return self.find_best_solution_dfs()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--puzzle', default='mrf-sna-opu-gci', type=str, help='puzzle input in abd-def-ghi-jkl format')
    parser.add_argument('--dict', default='words.txt', type=str, help='path to newline-delimited text file of valid words')
    parser.add_argument('--astar', action='store_true', help='Use A* search instead of DFS')
    args = parser.parse_args()

    print("solving puzzle", args.puzzle)
    puzzle = LetterBoxed(args.puzzle, args.dict)
    print(len(puzzle.puzzle_words), "valid words found")
    best_solution = puzzle.find_best_solution(args.astar)

    if best_solution:
        print("Best Solution (fewest words):", best_solution)
        print("Number of words in best solution:", len(best_solution))
    else:
        print("No solution found.")