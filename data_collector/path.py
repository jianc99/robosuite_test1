from collections import deque
import numpy as np

def bfs(grid, start, target):
    rows = len(grid)
    cols = len(grid[0])
    visited = [[False] * cols for _ in range(rows)]
    path = [[None] * cols for _ in range(rows)]
    queue = deque([start])
    visited[start[0]][start[1]] = True

    while queue:
        current = queue.popleft()
        if current == target:
            break

        row, col = current
        neighbors = [(row-1, col), (row+1, col), (row, col-1), (row, col+1)]

        for neighbor in neighbors:
            nrow, ncol = neighbor
            if 0 <= nrow < rows and 0 <= ncol < cols and grid[nrow][ncol] == 0 and not visited[nrow][ncol]:
                queue.append((nrow, ncol))
                visited[nrow][ncol] = True
                path[nrow][ncol] = current

    if not visited[target[0]][target[1]]:
        return None  # No path found

    # Reconstruct the path with fewer waypoints
    path_list = []
    current = target
    while current != start:
        path_list.append(current)
        current = path[current[0]][current[1]]
    path_list.append(start)
    path_list.reverse()

    # Simplify the path to include only turning points
    simplified_path = [path_list[0]]
    for i in range(1, len(path_list) - 1):
        prev_row, prev_col = simplified_path[-1]
        curr_row, curr_col = path_list[i]
        next_row, next_col = path_list[i + 1]
        if (prev_row - curr_row != curr_row - next_row) or (prev_col - curr_col != curr_col - next_col):
            simplified_path.append(path_list[i])
    simplified_path.append(path_list[-1])

    return simplified_path


# grid = [
#     [0, 0, 0, 0, 0, 0],
#     [0, 1, 1, 0, 1, 1],
#     [1, 0, 0, 0, 0, 0],
#     [0, 0, 0, 1, 1, 0]
# ]

# start_pos = (0, 0)
# target_pos = (3, 5)

def find_path(grid, start_pos, target_pos):
    path = bfs(grid, start_pos, target_pos)
    if path is None:
        print("No path found")
        return None
    else:
        # Simplify the path further to include only a few waypoints
        simplified_path = [path[0]]
        for i in range(1, len(path) - 1):
            prev_row, prev_col = simplified_path[-1]
            curr_row, curr_col = path[i]
            next_row, next_col = path[i + 1]
            if (prev_row - curr_row != curr_row - next_row) and (prev_col - curr_col != curr_col - next_col):
                simplified_path.append(path[i])
        simplified_path.append(path[-1])
        print("Path found:", simplified_path)
        return simplified_path[1:]

