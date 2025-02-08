# DollarGame
Python implementation of the "Dollar Game" as seen on [Numberphile](https://www.youtube.com/watch?v=U33dsEcKgeQ&ab_channel=Numberphile).

The object of the game is to eliminate all negative numbers from the graph.  This is done by redistributing the "dollars" on each node. When you click a node, one dollar will be given to each neighbor of that node, possibly causing it to go negative itself.  Keep this up until you've moved enough dollars around to eliminate all debt.  

To play the game, clone the repo and type "python game.py" (or py game.py or python3 game.py or whatever you use locally). You will be prompted for a desired number of vertices, followed by a difficulty.  All games are beatable, so no complaining about the difficulty!

In the video above, a specific challenge was presented. That graph can be found [here](https://www.bradyharanblog.com/blog/the-dollar-game). In a separate file, you can play either the regular Dollar Game, or this specific challenge.  That file is cleverly named game_with_numberphile.py because I'm a naming genius. You will be prompted with the option to play the regular game, or the numberphile challenge. I make no claim on the beatability of the "monster", but I tend to trust Prof. Krieger on this one.
