import cv2
import mediapipe as mp
import numpy as np
import random

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(model_complexity=1,max_num_hands=1,min_tracking_confidence=0.8,min_detection_confidence=0.8)
mp_drawing = mp.solutions.drawing_utils


# Game settings
width, height = 1280, 640
player_pos = [320, 440]
# enemy speed, size, and list initialization
enemy_size = 40
enemy_speed = 20
enemy_list = []

# Initialize score
score = 0

# Create random enemy
def create_enemy():
    x_pos = random.randint(0,width-enemy_size)
    return[x_pos,0]    

# Move enemies down
def move_enemies(enemy_list):
    for enemy in enemy_list:
        enemy[1] += enemy_speed
# Check if enemy is off-screen
# Increment score for each enemy that goes off-screen
def remove(enemy_list):
    global score 
    new_list = []
    for enemy in enemy_list:
        if enemy[1]>height:
            score +=1
        else:
            new_list.append(enemy)
    return new_list            


# Check for collisions
def check_collision(player_pos, enemy_list):
    player_x ,player_y = player_pos
    for enemy in enemy_list:
        enemy_x , enemy_y = enemy
        if (enemy_x - enemy_size<player_x<enemy_x + enemy_size and enemy_y-enemy_size<player_y<enemy_y+enemy_size):
            return True
        else:
            return False

#testing things idk what it is
previous_x, previous_y = None, None
smooth_factor = 0.9
# Initialize webcam
cap = cv2.VideoCapture(0)


while True:
    ret, frame = cap.read()
    
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame with MediaPipe
    result = hands.process(rgb_frame)

            
    # Get coordinates of the index finger tip (landmark 8)
    height1,width1,_=frame.shape



    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            index = hand_landmarks.landmark[8]
            player_pos[0]=int(index.x*width)
            player_pos[1]=int(index.y*height)
    # trying to do something and make it better idk what how
        if previous_x is not None and previous_y is not None:
                player_pos[0] = int(previous_x * smooth_factor + player_pos[0] * (1 - smooth_factor))
                player_pos[1] = int(previous_y * smooth_factor + player_pos[1] * (1 - smooth_factor))
            
            #idk why sommething from chatgpt
        previous_x, previous_y = player_pos[0], player_pos[1]
            

    # Add new enemies
    if random.randint(0,30)<5:
        enemy_list.append(create_enemy())
    
    # Move enemies
    move_enemies(enemy_list)
    
    # Check for collision
    if check_collision(player_pos,enemy_list):
        print(f"Game over {score}")
        break
    enemy_list=remove(enemy_list)
    # Draw game elements
    cv2.circle(frame,(player_pos[0],player_pos[1]),30,(0,255,0),-1)
    for enemy in enemy_list:
        cv2.rectangle(frame,(enemy[0],enemy[1]),(enemy[0]+enemy_size,enemy[1]+enemy_size),(0,0,255),-1)
    
    # Display score on the frame
    cv2.putText(frame,f"score = {score}",(10,50),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(255,0,0),5)

    cv2.imshow("Object Dodging Game", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
