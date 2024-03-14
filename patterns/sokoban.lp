all_boxes_immoveable :- #count{X, Y: box_stuck(X,Y)} = 4.
box_of_boxes :- box(X,Y), box(X+1,Y), box(X,Y-1), box(X+1,Y-1).
agent_box_corners :- agent(X,Y), box(X+1,Y+1), box(X+1,Y-1), box(X-1,Y+1), box(X-1,Y-1).
