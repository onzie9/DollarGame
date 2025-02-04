import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random


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
        draw_graph()


# Function to draw the graph with current values
def draw_graph():
    ax.clear()  # Clear the previous plot

    # Draw the graph
    nx.draw(G, pos, ax=ax,
            #with_labels=True,
            node_size=2000,
            node_color="skyblue",
            font_size=15,
            font_weight="bold")

    # Add node values to the graph
    for node, (x, y) in pos.items():
        ax.text(x, y, str(node_values[node]), fontsize=12, ha='center', va='center')

    if winning_graph(node_values):
        ax.set_title("You win!")
    else:
        ax.set_title("Dollar Game")
    plt.draw()


def winning_graph(nodes: dict) -> bool:
    if all(x>=0 for x in nodes.values()):
        return True
    return False

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
draw_graph()

# Show the plot
plt.show()