import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.art3d as art3d

import matplotlib
# Define pose positions (y, z coordinates)
pose_dictionary = {
    0: [-85, 271], 1: [0, 271], 2: [85, 271],
    3: [-85, 192], 4: [0, 192], 5: [85, 192],
    6: [-85, 123], 7: [0, 123], 8: [85, 123]
}
BOARD_POSITION=7
# Define the motion along the x-axis (towards and away from the camera)
forward_motion = 800  # Forward 25 mm along the x-axis
backward_motion = 550  # Backward 25 mm along the x-axis

# ChArUco board dimensions
size_of_checkerboard = 25  # Size of each checkerboard square in mm
aruco_w = 7  # Number of columns
aruco_h = 5  # Number of rows

# Calculate the real-world dimensions of the ChArUco board
board_width = aruco_w * size_of_checkerboard  # 7 squares width * 25 mm per square
board_height = aruco_h * size_of_checkerboard  # 5 squares height * 25 mm per square

# A5 148 x 210
board_width = 148
board_height = 210

# Position of the board: Place it at position 4 (middle of the grid, y=0, z=192)
board_center_y, board_center_z = pose_dictionary[BOARD_POSITION]

# Calculate the corners of the ChArUco board, making sure the center is at P4
half_width = board_width / 2
half_height = board_height / 2

board_corners_y = [
    board_center_y - half_width, board_center_y + half_width,
    board_center_y + half_width, board_center_y - half_width
]
board_corners_z = [
    board_center_z - half_height, board_center_z - half_height,
    board_center_z + half_height, board_center_z + half_height
]
board_x_position = 150+backward_motion
board_corners_x = [board_x_position] * 4


# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# Plot each position with forward and backward motion along the x-axis
for i, (y, z) in pose_dictionary.items():
    # Plot forward and backward motion points
    ax.scatter(forward_motion, y, z, color='green', s=100)  # Forward position --- , label='Forward' if i == 0 else ""
    ax.scatter(backward_motion, y, z, color='red', s=100)  # Backward position ---- , label='Backward' if i == 0 else ""

    # Draw a line between forward and backward points
    ax.plot([backward_motion, forward_motion], [y, y], [z, z], color='blue')

    # Annotate the positions slightly offset from the points
    ax.text(forward_motion, y+15, z, f'P{i}', fontsize=10, color='black', ha='center')
    #ax.text(backward_motion - 15, y, z,  fontsize=10, color='black', ha='center') #f'P{i}',

# add dotted lines between points 0-1, 1-2, 3-4, 4-5, 6-7, 7-8 only in forward motion
#for i in [0, 1, 3, 4, 6, 7]:
#    ax.plot([forward_motion, forward_motion], [pose_dictionary[i][0], pose_dictionary[i+1][0]], [pose_dictionary[i][1], pose_dictionary[i+1][1]], color='black', linestyle='--')


# Plot the ChArUco board as a rectangle on the y-z plane at position 4 (P4)
ax.plot(board_corners_x + [board_corners_x[0]],  # Close the rectangle
        board_corners_y + [board_corners_y[0]],
        board_corners_z + [board_corners_z[0]],
        color='black', linewidth=2) #, label='ChArUco Board')

# fill in area inside the rectangle
for i in ["x"]:
    rect = plt.Rectangle((board_corners_y[0], board_corners_z[0]), board_width, board_height, color='lightgray', alpha=0.01)
    ax.add_patch(rect)
    art3d.pathpatch_2d_to_3d(rect, z=0, zdir=i)

#matplotlib.patches.Rectangle((board_corners_y[0], board_corners_z[0]), board_width, board_height, color='lightgray', alpha=0.01)


""" ax.fill(board_corners_x + [board_corners_x[0]],  # Close the rectangle
        board_corners_y + [board_corners_y[0]],
        board_corners_z + [board_corners_z[0]],
        color='lightgray', alpha=0.01) """

# Annotate the ChArUco board's dimensions
""" ax.text(0, board_center_y, 5, f'ChArUco Board\n{board_width}mm x {board_height}mm',
        fontsize=10, color='black', ha='center') """
        
# Set plot labels and title
# Set plot labels and title
ax.set_title(f"Robot positions in 3D with ChArUco Board at P{BOARD_POSITION}")
ax.set_xlabel("X (mm)") #Depth from camera in 
ax.set_ylabel("Y (mm)")
ax.set_zlabel("Z (mm)")
# make x ticks less dense but keeo firs and last
ax.set_xticks([backward_motion, forward_motion])
# remove grid lines
ax.grid(False)
# axis equal
#ax.set_aspect('equal')

# Add a legend to indicate forward and backward motion
#ax.legend(loc='upper right')

# Show the plot
# move axis to 45 degrees
ax.view_init(elev=13, azim=-17)
plt.savefig('robot_positions_with_charuco_board.pdf')
plt.show()
