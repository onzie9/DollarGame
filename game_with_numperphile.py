import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import pandas as pd
from PIL import Image


def generate_random_integers(n: int, g:int) -> list[int]:
    # Generate N-1 random integers between -9 and 9
    result = [random.randint(-9, 9) for _ in range(n - 1)]

    # Calculate the last integer to ensure the sum is equal to g
    last_integer = g - sum(result)

    # Ensure the last integer is within the range of -9 to 9
    if last_integer < -9 or last_integer > 9:
        # If not, recursively try again
        return generate_random_integers(n, g)

    # Append the last integer to the list
    result.append(last_integer)

    return result


def generate_random_connected_graph(n):
    if n < 2:
        raise ValueError("Number of vertices must be at least 2 to create a connected graph.")

    # Start with an empty graph
    G = nx.Graph()

    # Add n vertices to the graph
    G.add_nodes_from(range(n))

    # Step 1: Create a spanning tree (this guarantees the graph is connected)
    # We can start by adding a chain of edges (a simple spanning tree)
    for i in range(n - 1):
        G.add_edge(i, i + 1)

    # We add edges while ensuring they don't create self-loops or duplicate edges
    possible_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    random.shuffle(possible_edges)

    # Adding extra edges randomly, not exceeding the number of possible edges
    num_extra_edges = random.randint(0, len(possible_edges) // 2)

    for i in range(num_extra_edges):
        u, v = possible_edges[i]
        G.add_edge(u, v)
    return G


def graph_genus(graph: nx.classes.graph.Graph) -> int:
    return graph.size() - graph.order() + 1


# Function to update the node values after a click event
def on_click(event):
    # Get the mouse position
    mouse_x, mouse_y = event.xdata, event.ydata
    if mouse_x is None or mouse_y is None:
        return  # Ignore if click is outside the plot area

    # Find the closest node
    closest_node = None
    min_distance = float('inf')
    for node, (x, y) in pos.items():
        dist = np.sqrt((mouse_x - x) ** 2 + (mouse_y - y) ** 2)
        if dist < min_distance:
            min_distance = dist
            closest_node = node

    # Check the degree of the node and update values
    if closest_node is not None:
        # Get the degree of the clicked node
        node_degree = G.degree[closest_node]

        # Decrease the value of the clicked node by its degree
        node_values[closest_node] -= node_degree

        # Increase the value of its neighbors by 1
        for neighbor in G.neighbors(closest_node):
            node_values[neighbor] += 1

        # Redraw the graph with updated values
        draw_graph(G=G, node_vals_var=node_values, ns=ns)


# Function to draw the graph with current values
def draw_graph(num_phile: bool = False,
               npinf: bool = False,
               monster_graph: nx.Graph = None,
               G: nx.Graph = None,
               node_vals_var: dict = None,
               np_initial_nodes: dict = None,
               ns: int = None):

    ax.clear()  # Clear the previous plot

    # Draw the graph
    if num_phile:
        G = monster_graph
        if npinf:
            node_values = np_initial_nodes
            npinf = False

    nx.draw(G, pos, ax=ax,
            node_size=ns,
            node_color="skyblue",
            font_size=15,
            font_weight="bold")

    # Add node values to the graph
    for node, (x, y) in pos.items():
        ax.text(x, y, str(node_vals_var[node]), fontsize=12, ha='center', va='center')

    if winning_graph(node_vals_var):
        ax.set_title("You win!")
        plt.savefig("Winning graph.")
        print("You won the game! The winning graph has been saved as a png.")
        plt.close()
    else:
        ax.set_title("Dollar Game")
    plt.draw()


def winning_graph(nodes: dict) -> bool:
    if all(x>=0 for x in nodes.values()):
        return True
    return False

npinf = False
num_phile = input("Do you want to play the Numberphile monster graph? "
                  "(You might need to adjust your window size. (y/n) ")
if num_phile == 'y':
    num_phile = True
    npinf = True
    ns = 500
    # Read image.
    img = cv2.imread('dollar_monster_bw.jpg', cv2.IMREAD_COLOR)

    # Convert to grayscale.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Blur using 3 * 3 kernel.
    gray_blurred = cv2.blur(gray, (3, 3))

    # Apply Hough transform on the blurred image.
    detected_circles = cv2.HoughCircles(gray_blurred,
                                        cv2.HOUGH_GRADIENT, 1, 20, param1=50,
                                        param2=30, minRadius=1, maxRadius=40)

    # Draw circles that are detected.
    if detected_circles is not None:

        # Convert the circle parameters a, b and r to integers.
        detected_circles = np.uint16(np.around(detected_circles))

        for pt in detected_circles[0, :]:
            a, b, r = pt[0], pt[1], pt[2]

    img = Image.open('dollar_monster_bw.jpg')
    width = img.width
    height = img.height
    circles_df = pd.DataFrame(columns=['x', 'y', 'r'], data=detected_circles[0])

    x_y_points = {i: [float(circles_df['x'].iloc[i]), float(circles_df['y'].iloc[i])] for i in range(len(circles_df))}
    manual_node_start = len(x_y_points)
    x_y_points[manual_node_start] = [310, 75]
    x_y_points[manual_node_start + 1] = [250, 765]
    x_y_points[manual_node_start + 2] = [1670, 535]
    x_y_points[manual_node_start + 3] = [1650, 795]

    monster_graph = nx.Graph()


    #pos = nx.rescale_layout_dict(x_y_points)
    monster_graph.add_nodes_from(range(len(x_y_points)))

    edges = [(29, 40),
             (29, 9),
             (29, 63),
             (9, 73),
             (73, 84),
             (73, 44),
             (73, 66),
             (73, 61),
             (40, 84),
             (40, 63),
             (63, 41),
             (63, 44),
             (44, 66),
             (66, 20),
             (66, 57),
             (66, 3),
             (17, 84),
             (17, 43),
             (17, 51),
             (17, 61),
             (84, 51),
             (61, 51),
             (61, 3),
             (51, 26),
             (26, 76),
             (41, 20),
             (20, 7),
             (84, 66),
             (50, 22),
             (50, 76),
             (50, 81),
             (57, 0),
             (57, 3),
             (57, 74),
             (51, 76),
             (0, 32),
             (0, 46),
             (32, 7),
             (32, 78),
             (7, 83),
             (7, 78),
             (83, 54),
             (54, 42),
             (42, 25),
             (25, 74),
             (74, 47),
             (74, 60),
             (78, 54),
             (78, 42),
             (32, 46),
             (60, 13),
             (60, 42),
             (60, 47),
             (60, 64),
             (64, 78),
             (54, 64),
             (13, 64),
             (64, 74),
             (74, 81),
             (57, 81),
             (81, 72),
             (81, 76),
             (72, 74),
             (57, 74),
             (34, 81),
             (34, 76),
             (76, 4),
             (4, 22),
             (22, 69),
             (15, 69),
             (12, 15),
             (15, 30),
             (11, 30),
             (30, 35),
             (4, 12),
             (11, 18),
             (11, 19),
             (18, 35),
             (19, 35),
             (19, 45),
             (19, 56),
             (19, 80),
             (31, 80),
             (49, 80),
             (35, 80),
             (24, 31),
             (24, 86),
             (24, 39),
             (24, 67),
             (18, 6),
             (18, 39),
             (52, 12),
             (52, 30),
             (52, 45),
             (36, 12),
             (36, 79),
             (56, 67),
             (56, 45),
             (79, 81),
             (72, 81),
             (79, 62),
             (79, 33),
             (79, 48),
             (14, 47),
             (14, 82),
             (14, 33),
             (48, 82),
             (48, 58),
             (48, 64),
             (28, 58),
             (16, 28),
             (16, 70),
             (58, 82),
             (16, 72),
             (57, 72),
             (33, 36),
             (33, 62),
             (52, 62),
             (52, 53),
             (53, 68),
             (53, 36),
             (53, 36),
             (53, 37),
             (53, 45),
             (53, 62),
             (62, 67),
             (56, 65),
             (24, 37),
             (65, 67),
             (6, 38),
             (21, 1),
             (21, 33),
             (21, 70),
             (21, 59),
             (16, 71),
             (28, 70),
             (8, 70),
             (8, 59),
             (1, 59),
             (27, 59),
             (65, 68),
             (65, 71),
             (65, 27),
             (65, 2),
             (65, 38),
             (38, 49),
             (38, 55),
             (2, 71),
             (2, 55),
             (10, 55),
             (5, 23),
             (5, 75),
             (23, 75),
             (49, 39),
             (49, 71),
             (49, 77),
             (75, 77),
             (75, 85),
             (77, 85),
             (10, 75),
             (71, 77)]
    for e in edges:
        monster_graph.add_edge(e[0], e[1])
    pos = nx.spring_layout(monster_graph)

    for k in x_y_points.keys():
        p = x_y_points[k]
        x_y_points[k] = [p[0] - width, height - p[1]]
        pos[k] = np.array([p[0] - width, height - p[1]])

    node_vals = pd.read_csv('MonsterDollarNodes.csv', header=0)

    np_initial_nodes = {}
    for i in range(len(node_vals)):
        np_initial_nodes[i] = int(node_vals['value'].iloc[i])
    G = monster_graph
    node_values = np_initial_nodes

else:
    num_phile = False
    npinf = False
    monster_graph = None
    np_initial_nodes = None
    ns = 2000
    n = input("How many nodes would you like? ")
    n = int(n)
    dif = input("What level of difficulty would you like? (0-5) ")
    dif = int(dif)
    # Create a graph

    G = generate_random_connected_graph(n)
    g = graph_genus(G) + 5 - dif
    nodes = generate_random_integers(n, g)
    node_values = {i: nodes[i] for i in range(n)}

    # Positioning for nodes in 2D space
    pos = nx.spring_layout(G)

# Plotting the graph
fig, ax = plt.subplots(figsize=(6, 6))

# Connect the click event to the on_click function
fig.canvas.mpl_connect('button_press_event', on_click)

# Initial draw of the graph

draw_graph(num_phile, npinf, monster_graph, G, node_values, np_initial_nodes, ns)

# Show the plot
plt.show()