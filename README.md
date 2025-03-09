

# LetterBoxed Solver


*   **Trie (Prefix Tree):** A tree-like data structure used to efficiently store and search for words based on their prefixes.  This significantly speeds up the process of finding valid words during the search.
*   **Bitmasks:**  An integer representation where each bit corresponds to a specific letter from the puzzle.  This allows for efficient checking of which letters have been used and whether a word covers new letters.
*   **Graph Representation:**  The valid words are organized into a graph where nodes represent letters, and edges represent words that connect a starting letter to an ending letter.  The edges are further categorized by the set of letters they use (represented as a bitmask).
*   **Depth-First Search (DFS):** A recursive search algorithm that explores as far as possible along each branch before backtracking.
*   **A* Search:** An informed search algorithm that uses a heuristic function to estimate the cost of reaching the goal from a given state. This helps prioritize exploring the most promising paths, potentially finding solutions faster than DFS.
*   **Heuristic Function:** In this implementation, the heuristic estimates the minimum number of words needed to cover the remaining letters. It's based on the maximum number of new letters that can be covered by a single word from the current state.

## Classes

### `WordTrieNode`

Represents a node in the Trie.

*   `value`: The letter stored at this node.
*   `parent`: The parent node (or `None` for the root).
*   `children`: A dictionary mapping letters to child nodes.
*   `valid`: A boolean indicating whether the path from the root to this node forms a valid word.
*   `get_word()`:  Returns the full word represented by the path from the root to this node.

### `LetterBoxed`

The main class that handles the puzzle logic.

*   `__init__(input_string, dictionary)`:
    *   Initializes the puzzle with the input string (sides of the box) and the dictionary file.
    *   Creates the Trie by adding all words from the dictionary.
    *   `input_string`: The input string defining the puzzle (e.g., "abc-def-ghi-jkl").
    *   `sides`: A set of the four sides of the puzzle.
    *   `puzzle_letters`: A set of all unique letters in the puzzle.
    *   `letter_to_bitmask`: A dictionary mapping each letter to a unique bitmask.
    *   `all_letters_mask`:  A bitmask representing all 12 letters being used.
    *   `root`: The root node of the Trie. (Which is usually insignigicant)
    *   `puzzle_words`: A list of valid words that can be formed using the letters in the puzzle. Pre-filtered to include only words that are potentially part of an optimal solution.
    *    `puzzle_graph`: A nested defaultdict representing a graph.  `puzzle_graph[start_letter][end_letter][letter_mask]` stores a list of words starting with `start_letter`, ending with `end_letter` and covering the letters indicated by the `letter_mask`.

*   `add_word(word)`: Adds a word to the Trie.

*   `_puzzle_words_inner(node, last_side)`:  A recursive helper function to find all valid words starting from a given Trie node and avoiding letters from the `last_side`.

*   `get_puzzle_words()`:  Finds all valid words that can be used in the puzzle and pre-filters them.  The pre-filtering is crucial: It removes words that are "useless" because all the letters they contain are already covered by another, shorter, word.

*   `_find_solutions_inner(path_words, letters_mask, next_letter, max_len)`:  A recursive helper function for the DFS algorithm.  It searches for solutions given a current path, the letters used so far, the next letter to start from, and a maximum path length.

*   `find_best_solution_dfs()`: Implements the Depth-First Search algorithm to find the best solution.

*   `heuristic(letters_mask)`:  Calculates the heuristic value (estimated remaining words) for the A* search.

*   `find_best_solution_astar()`: Implements the A* search algorithm.

*   `find_best_solution(use_astar=False)`:  A wrapper function that calls either `find_best_solution_astar` or `find_best_solution_dfs` based on the `use_astar` flag.

## How It Works (DFS)

1.  **Initialization:** The `LetterBoxed` class initializes the puzzle, builds the Trie, and creates the `puzzle_graph`.
2.  **Word Filtering:** `get_puzzle_words()` identifies all valid words from the dictionary that can be formed using the puzzle's letters and are *useful* (see pre-filtering above).
3.  **Graph Construction:**  The `puzzle_graph` is built, connecting letters with words (represented as edges).
4.  **DFS Search:** `find_best_solution_dfs()` starts the search:
    *   It iterates through possible maximum solution lengths (from 1 to 5).
    *   For each length, it tries all possible starting letters.
    *   `_find_solutions_inner()` recursively explores the graph:
        *   If all letters are used (`letters_mask == self.all_letters_mask`), a solution is found.
        *   If the maximum path length is reached, the branch is abandoned.
        *   Otherwise, it explores all possible next words from the current letter, recursively calling itself.
    *   The shortest solution found is returned.

## How It Works (A*)

1.  **Initialization:**  Similar to DFS, initializes the puzzle, Trie, and `puzzle_graph`.
2.  **Word Filtering:** Identical to DFS.
3.  **Graph Construction:** Identical to DFS.
4.  **A* Search:** `find_best_solution_astar()` uses a priority queue (`open_set`) to manage the search:
    *   Initial states (starting with each possible first word) are added to the priority queue.  The priority is determined by the f-score (g-score + h-score).
    *   The algorithm repeatedly extracts the state with the lowest f-score from the queue.
    *   **Visited Set:** A `visited` set is used to avoid revisiting the same states.
    *   **Goal Check:** If all letters are used, a solution is found. The search terminates immediately, as A* guarantees finding the optimal solution first.
    *   **Expansion:**  If the goal isn't reached, the algorithm expands the current state by considering all possible next words.
    *   **Heuristic:** The heuristic function (`heuristic()`) estimates the remaining cost (number of words) to reach the goal.
    *   **Priority Queue Update:** New states are added to the priority queue with their corresponding f-scores.
    * **Pruning:**  If a potential path's f-score (g-score + heuristic) is greater than or equal to the length of the best solution found so far, that path is pruned (discarded), greatly improving efficiency.
5. **Result:** The algorithm returns as soon as it finds a solution (which is guaranteed to be optimal). If the queue becomes empty, no solution exists.

## Key Improvements and Optimizations

*   **Trie:** Using a Trie drastically reduces the time spent checking if a word is valid.
*   **Bitmasks:**  Bitwise operations are very fast, making letter checking and tracking highly efficient.
*   **Pre-filtering of `puzzle_words`:** This significantly reduces the search space by removing words that cannot contribute to an optimal solution.
*   **Graph Representation (`puzzle_graph`):** This pre-computed graph allows for quick lookup of possible next words.
*   **Heuristic Function (A*):** The heuristic guides the search towards more promising paths, leading to faster convergence to the solution.
*   **Early Pruning (A*):** By keeping track of the best solution length found so far, the A* search can prune branches that are guaranteed to be worse, significantly improving performance.
* **Product from itertools:** Efficiently generate all possible pairs.

## Usage

```bash
python main.py

DFS (default)
python main.py --puzzle peh-unt-qir-kas
AStar
python main.py --puzzle peh-unt-qir-kas --astar