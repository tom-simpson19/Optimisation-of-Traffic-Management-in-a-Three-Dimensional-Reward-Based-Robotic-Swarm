from vpython import *
import random
import math
import matplotlib.pyplot as plt

random.seed(40)

#######################################
'''
This code was written and tested using 
python 3.8.0.
vpython 7.6.1
'''
#######################################

# Create ball and define radius
def ball_pos(ball, first_target):
    #Initilise y position of objects to a half side length
    ball.pos.y = random.randrange(-int(0.5 * side), int(0.5 * side))
    #Avoid any zero errors
    while ball.pos.y == 0:
        ball.pos.y = random.randrange(-int(side - 1), int(side - 1))
    ball.pos.x = random.randrange(-int(abs(ball.pos.y) / tan(0.25 * pi)), int(abs(ball.pos.y) / tan(0.25 * pi)))
    ball.pos.x = random.randrange(-int(abs(ball.pos.y) / tan(0.25 * pi)), int(abs(ball.pos.y) / tan(0.25 * pi)))
    #Initiliase direction vector
    ball.p = vector(0 - ball.pos.x, 0 - ball.pos.y, 0 - ball.pos.z)
    while True:
        # Define all ball to be in top or bottom pyramids
        if abs(ball.pos.x) <= abs(ball.pos.y) * math.tan(pi * 0.25):
            if abs(ball.pos.z) <= abs(ball.pos.y) * math.tan(pi * 0.25):
                break
        else:
            ball.pos.x = random.randrange(-int(abs(ball.pos.y) / tan(0.25 * pi)), int(abs(ball.pos.y) / tan(0.25 * pi)))
            ball.pos.x = random.randrange(-int(abs(ball.pos.y) / tan(0.25 * pi)), int(abs(ball.pos.y) / tan(0.25 * pi)))

    #Define first target positions
    first_target.pos.y = random.randrange(int(- (side - 1)), int(side - 1))
    first_target.pos.x = random.randrange(-int(side - 1), int(side - 1))
    first_target.pos.z = random.randrange(-int(side - 1), int(side - 1))
    
    #Definie x position of object same quarter as target
    if first_target.pos.x > 0:
        ball.pos.x = abs(ball.pos.x)
        ball.p.x = abs(ball.p.x)

    #Definie x position of object same quarter as target
    elif first_target.pos.x <= 0:
        ball.pos.x = -abs(ball.pos.x)
        ball.p.x = -abs(ball.p.x)

    #Definie y position of object same quarter as target
    if first_target.pos.y > 0:
        ball.pos.y = abs(ball.pos.y)
        ball.p.y = abs(ball.p.y)

    #Definie y position of object same quarter as target
    elif first_target.pos.y <= 0:
        ball.pos.y = -abs(ball.pos.y)
        ball.p.y = -abs(ball.p.y)

    #Definie z position of object same quarter as target
    if first_target.pos.z > 0:
        ball.pos.z = abs(ball.pos.z)
        ball.p.z = abs(ball.p.z)
    
    #Definie z position of object same quarter as target
    elif first_target.pos.z <= 0:
        ball.pos.z = -abs(ball.pos.z)
        ball.p.z = -abs(ball.p.z)

    #Initialise state as normal
    ball.state = "normal"
    return ball, first_target


def hit_target(ball, first_target, allobjects, index):
    #Keep original target position
    original_target_pos = first_target.pos
    #keep original ball position
    original_ballpos = ball.pos
    #Keep original ball direction vector
    original_direction = ball.p
    #define new target position
    first_target.pos.y = random.randrange(int(- (side - 1)), int(side - 1))
    determine_negtaive = random.randint(1, 3)
    first_target.pos.x = random.randrange(-int(side - 1), int(side - 1))
    first_target.pos.z = random.randrange(-int(side - 1), int(side - 1))
    #Set ball position as "exited area"
    ball.pos = vector(0, 1.5, 0)
    #Define new object direction vector based off target position
    ball.p = vector(first_target.pos.x * 0.1, first_target.pos.y * 0.5, first_target.pos.z * 0.1) - ball.pos

    #Definie x position of object same quarter as target
    if first_target.pos.x >= 0:
        ball.pos.x = abs(ball.pos.x)

    #Definie x position of object same quarter as target
    elif first_target.pos.x < 0:
        ball.pos.x = -abs(ball.pos.x)

    #Definie y position of object same quarter as target
    if first_target.pos.y >= 0:
        ball.pos.y = abs(ball.pos.y)

    #Definie y position of object same quarter as target
    elif first_target.pos.y < 0:
        ball.pos.y = -abs(ball.pos.y)

    #Definie z position of object same quarter as target
    if first_target.pos.z >= 0:
        ball.pos.z = abs(ball.pos.z)

    #Definie z position of object same quarter as target
    elif first_target.pos.z < 0:
        ball.pos.z = -abs(ball.pos.z)

    #Iterate through all objects
    for i in range(len(allobjects)):
    	if not i == index:
            #Calculate distance between objects
    		distance =dist_calculator(ball.pos, allobjects[i].pos)
            #If objects colliding 
    		if distance <= 2* radius_ball:
                #Set ball back to original position
    			ball.pos = original_ballpos
                #Set state to waiting 
    			ball.state = "waiting"
                #Set target position back to original
    			first_target.pos = original_target_pos
    			return ball, first_target.pos
    #Else state to normal
    ball.state = "normal"
    #Add one to object score 
    ball.score += 1
    #reset target condition
    ball.target_condition = "not_hit"
    return ball, first_target.pos


def normalise_vector(p):
    p_norm = vector(0, 0, 0)
    # Clculate some of absolute values original p vector
    sum_old = abs(p.x) + abs(p.y) + abs(p.z)
    # Calculate normalised values
    if not sum_old == 0:
        p_norm.x, p_norm.y, p_norm.z = p.x / sum_old, p.y / sum_old, p.z / sum_old

        return p_norm
    else:
        return p


def test_cutoff(ball1, dist_ceneter, condition1, condition2, condition3):
    number_in_cutoff = 0
    #Calculate number in cutoff
    for l in range(len(ball)):
        if condition1:
            if condition2:
                if condition3:
                    dist_ceneter = sqrt((ball[l].pos.x) ** 2 + (ball[l].pos.y) ** 2 + (ball[l].pos.z) ** 2)
                    if dist_ceneter < cutoff_lower:
                        number_in_cutoff += 1
    #Calculate cutoff constraint
    if (number_in_cutoff * volume_ball) > (ratio * volume_cutoff):
        ball1.state = "waiting"
    else:
        ball1.state = "return"
    return ball1


def dist_calculator(obj1, obj2):
    distance = sqrt((obj1.x - obj2.x) ** 2 + (obj1.y - obj2.y) ** 2 + (obj1.z - obj2.z) ** 2)
    return distance


def avoid_collision(ball_i, ball_j):
    #Define scale factor 
    scale = abs(ball_i.p.x) + abs(ball_i.p.y) + abs(ball_i.p.z)
    #Avoid zero error
    if scale == 0:
        obj1.p = vector(0, 0, 0)
        return obj1.p
    #Normalise x,y,z 
    x, y, z = ball_i.p.x / scale, ball_i.p.y / scale, ball_i.p.z / scale

    if abs(x) > abs(y):
        temp1 = vector(z, 0, -x)
        temp1 = normalise_vector(temp1)
        temp2 = vector(-z, 0, x)
        temp2 = normalise_vector(temp2)
        ball_i.p = ball_i.p + temp1
        ball_j.p = ball_j.p + temp2
        return ball_i.p, ball_j.p
    else:
        temp1 = vector(0, z, -y)
        temp1 = normalise_vector(temp1)
        temp2 = vector(0, -z, y)
        temp2 = normalise_vector(temp2)
        ball_i.p = ball_i.p + temp1
        ball_j.p = ball_j.p + temp2
        return ball_i.p, ball_j.p


def normal_vector(vector_between):
    x = vector_between.x
    y = vector_between.y
    z = vector_between.z
    if vector_between == vector(-1, 1, 0):
        pre_normal_vector = vector(-y - z, -x, -x)
    else:
        pre_normal_vector = vector(z, z, -x - y)
    pre_normal_vector = normalise_vector(pre_normal_vector)
    return pre_normal_vector


def calclate_weighting(item, weighting, mean_score, current_score, mean_score_list, mean_reward_list, rerun, mode=1):
    # Add new mean score to the current list
    mean_score_list = mean_score_list + [current_score]
    # Calculate the new mean score
    mean_score = sum(mean_score_list) / (len(mean_score_list))
    
    if rerun != 0:
        if mode == 1:
            if current_score < (mean_score - ((1 - 0.975) * mean_score)):
                mean_reward_list[item] = mean_reward_list[item] + [-1]

            # if current score more than mean then a reward of 1
            elif current_score > (mean_score + ((1 - 0.975) * mean_score)):
                mean_reward_list[item] = mean_reward_list[item] + [1]

            # if current score equal to mean then a reward of 0
            elif (mean_score - ((1 - 0.975) * mean_score)) <= current_score <= (
                    mean_score + ((1 - 0.975) * mean_score)):
                mean_reward_list[item] = mean_reward_list[item] + [0]

        if mode == 2:
            mean_reward_list[item] = mean_reward_list[item] + [(current_score - mean_score) / mean_score]
        # Base weightng for each cutoff point on the reward
        weighting_temp[item] = weighting_temp[item] + sum(mean_reward_list[item]) / len(mean_reward_list[item])
        # Account for negative value, subtarct min so all >= 0
        min_weight = min(weighting_temp)
        max_weight = max(weighting_temp)

        if not min_weight == max_weight:
            if min_weight < 0:
                weighting_temp_adjusted = [(temp - min_weight) for temp in weighting_temp]
                # Calculate normaising factor
                if not sum(weighting_temp_adjusted) == 0:
                    normaliser = 1 / sum(weighting_temp_adjusted)
                    weighting = [(x * normaliser) for x in weighting_temp_adjusted]
            else:
                normaliser = 1 / sum(weighting_temp)
                weighting = [(x * normaliser) for x in weighting_temp]
        else:
            normaliser = 1 / sum(weighting_temp)
            weighting = [(x * normaliser) for x in weighting_temp]
        # Calculate normalised vector so sum = 1
        return weighting, mean_score, current_score, mean_score_list, mean_reward_list
    
    #Weight updates not done on first run
    elif rerun == 0:
        #Add current score to mean score
        mean_score_list = mean_score_list + [current_score]
        # Calculate the new mean score
        mean_score = sum(mean_score_list) / (len(mean_score_list))
        return weighting, mean_score, current_score, mean_score_list, mean_reward_list

def run_program(item, current_score):
    for k in range(number_itterations):
        for i in range(len(ball)):
            #Test for object existing outside the boundries
            if not (side > ball[i].pos.x > -side):
                ball[i].p.x = -ball[i].p.x

            if not (side > ball[i].pos.y > -side):
                ball[i].p.y = -ball[i].p.y

            if not (side > ball[i].pos.z > -side):
                ball[i].p.z = -ball[i].p.z

            #Normalise the direction vector 
            ball[i].p = normalise_vector(ball[i].p)
            #Calculate updated position vector
            ball[i].pos = ball[i].pos + (ball[i].p / ball[i].mass) * dt
            #Calculate direction vector from the center
            ball[i].p_center = (vector(0 - ball[i].pos.x, 0 - ball[i].pos.y, 0 - ball[i].pos.z))
            #Calculate the orginal position vector
            ball[i].pos_old = ball[i].pos - (ball[i].p / ball[i].mass) * dt
            #calculate the distance from the center
            dist_ceneter = sqrt((ball[i].pos.x) ** 2 + (ball[i].pos.y) ** 2 + (ball[i].pos.z) ** 2)
           
           #Account for issues with x equaling z
            while abs(ball[i].pos.x) == abs(ball[i].pos.z):
                ball[i].pos.x += 0.1

            #Redifne position states
            #If object has hit target then return
            if ball[i].target_condition == "hit":
                ball[i].state = "return"
            #If object has not hit target continue as normal
            if ball[i].target_condition == "not_hit":
                ball[i].state = "normal"

            for j in range(len(ball)):
                if ((i != j) and ball[j].state != "waiting"):
                    #Calculate distance between objects
                    dist_balls = sqrt((ball[i].pos.x - ball[j].pos.x) ** 2 + (ball[i].pos.y - ball[j].pos.y) ** 2 + (ball[i].pos.z - ball[j].pos.z) ** 2)
                    #If objects are colliding
                    if dist_balls <= 2 * radius_ball:
                        #Ball "remains" in original place
                        ball[i].pos = ball[i].pos_old
                        #Ball state is waiting
                        ball[i].state = "waiting"
                        vector_between = ball[j].p - ball[i].p
                        #Return vector that wont collide with opposing object
                        ball[j].p = normal_vector(vector_between)
                        break
            #Calculate distance from center
            dist = sqrt((ball[i].pos.x - 0) ** 2 + (ball[i].pos.y - 0) ** 2 + (ball[i].pos.z - 0) ** 2)

            #If object has reached center target
            if dist < radius_target:
                #Run hit target and return new position and target
                ball[i], first_target[i].pos = hit_target(ball[i], first_target[i], ball, i)

            # If pos of ball is hitting their target state = return
            dist_temp_target = dist_calculator(ball[i].pos, first_target[i].pos)
            if dist_temp_target < 2 * radius_ball:
                #Define state as return
                ball[i].state = "return"
                #Define target conditin as hit
                ball[i].target_condition = "hit"
                #add one to overall score
                current_score += 1

            # Test for position of ball being in top and bottom
            if abs(ball[i].pos.x) <= abs(ball[i].pos.y) * math.tan(pi * 0.25):
                if abs(ball[i].pos.z) <= abs(ball[i].pos.y) * math.tan(pi * 0.25):
                    # If first target is within this limit head towards it
                    if (abs(first_target[i].pos.x) <= abs(first_target[i].pos.y) * math.tan(pi * 0.25)):
                        if abs(first_target[i].pos.z) <= abs(first_target[i].pos.y) * math.tan(pi * 0.25):
                            #If object has hit its target
                            if ball[i].state == "return":
                                #Set p.y to zero so the object heads towards closest entrance region
                                ball[i].p.y = 0
                                #If direction vector is zero redifine to to p.center and set y to zero
                                if ball[i].p.x == ball[i].p.y == ball[i].p.z == 0:
                                    ball[i].p = ball[i].p_center
                                    ball[i].p.y = 0
                                if not (ball[i].p.x == 0) or (ball[i].p.z == 0):
                                    #Define +- based on current location to head towards a close point
                                    if abs(ball[i].pos.x) > abs(ball[i].pos.z):
                                        ball[i].p.z = 0
                                    elif abs(ball[i].pos.x) < abs(ball[i].pos.z):
                                        ball[i].p.x = 0
                                    elif abs(ball[i].pos.x) == abs(ball[i].pos.z):
                                        random_p = random.randint(1, 3)
                                        if random_p == 1:
                                            ball[i].p.x = 0
                                        elif random_p == 2:
                                            ball[i].p.y = 0

                            #If not hit target head towards it 
                            if ball[i].state != "return":
                                ball[i].p = first_target[i].pos - ball[i].pos

                    
                    # Upper limit contraint on object.
                    #If y values of object exceeds target move towards it
                    if abs(ball[i].pos.y) >= abs(first_target[i].pos.y):
                        ball[i].p = first_target[i].pos - ball[i].pos

            # Test for front and back
            if abs(ball[i].pos.z) >= abs(ball[i].pos.y) * math.tan(pi * 0.25):
                if abs(ball[i].pos.x) <= abs(ball[i].pos.z) * math.tan(pi * 0.25):
                   #If the object state is return then set direction vector to the center
                    if ball[i].state == "return":
                        ball[i].p = vector(0, 0, 0) - ball[i].pos
                        #If the ball position is within the convergence band test
                        if lower_limit[item] < dist_ceneter < cutoff_upper:
                            ball[i] = test_cutoff(ball[i], dist_ceneter, abs(ball[i].pos.z) >= abs(ball[i].pos.y) * math.tan(pi * 0.25), abs(ball[i].pos.x) <= abs(ball[i].pos.z) * math.tan(pi * 0.25), ball[i].pos.z > 0)
                    #If object is in normal state head towards the target
                    if ball[i].state == "normal":
                        ball[i].p = first_target[i].pos - ball[i].pos

            # Test for left and right
            if abs(ball[i].pos.z) <= abs(ball[i].pos.x) * math.tan(pi * 0.25):
                if abs(ball[i].pos.x) >= abs(ball[i].pos.y) * math.tan(pi * 0.25):
                    #If the object state is return then set direction vector to the center
                    if ball[i].state == "return":
                        ball[i].p = vector(0, 0, 0) - ball[i].pos
                    #If object is in normal state head towards the target
                    if ball[i].state == "normal":
                        ball[i].p = first_target[i].pos - ball[i].pos
                    #If the ball position is within the convergence band test
                    if lower_limit[item] < dist_ceneter < cutoff_upper:
                        ball[i] = test_cutoff(ball[i], dist_ceneter, abs(ball[i].pos.z) <= abs(ball[i].pos.x) * math.tan(pi * 0.25), abs(ball[i].pos.x) >= abs(ball[i].pos.y) * math.tan(pi * 0.25), ball[i].pos.x < 0)

    return current_score

outer_side = 15.0
#Length of cude
side = 10.0
#Define cutoff values
cutoff_upper = side / 3
cutoff_lower = side / 4
#calculate volume of cutoff
volume_cutoff = (4 / 3 * pi * (cutoff_lower ** 3)) / 6
#Refrences e greedy value
prob_best_upper = 0.8
#If defining a lower limit range specify here
lower_limit = [1.2, 1.8, 2.4]
#If selecting only one lower limit value keep this
lower_limit = [1.2]
# print(lower_limit)

radius_target = 1.0
#Alter radius of ball
radius_ball = 0.1
volume_ball = 4 / 3 * pi * (radius_ball ** 3)
ratio = 0.1

#Number itteration for lower limit
#number_itterations = 5000
#Number itteration for defined upper limit
number_itterations = 1000000
# Define inner wall boundries
pyramidR = pyramid(pos=vector(side, 0, 0), size=vector(side, 2 * side, 2 * side), color=color.red, opacity=(0.3)).rotate(angle=pi, axis=vec(0, 1, 0))
pyramidL = pyramid(pos=vector(-side, 0, 0), size=vector(side, 2 * side, 2 * side), color=color.blue, opacity=(0.3))
pyramidBack = pyramid(pos=vector(0, 0, -side), size=vector(side, 2 * side, 2 * side), color=color.yellow, opacity=(0.3)).rotate(angle=-pi / 2, axis=vec(0, 1, 0))
pyramidBottom = pyramid(pos=vector(0, -side, 0), size=vector(side, 2 * side, 2 * side), color=color.orange, opacity=(0.3)).rotate(angle=pi / 2, axis=vec(0, 0, 1))
pyramidTop = pyramid(pos=vector(0, side, 0), size=vector(side, 2 * side, 2 * side), color=color.green, opacity=(0.3)).rotate(angle=-pi / 2, axis=vec(0, 0, 1))

number_balls = 20
ball = [0] * number_balls
first_target = [0] * number_balls
for i in range(len(ball)):
    ball[i] = sphere(color=color.green, radius=radius_ball, make_trail=True, retain=200)
    first_target[i] = sphere(color=color.yellow, radius=radius_ball, make_trail=False, retain=200)
    ball[i].mass = 1.0
    ball[i].state = "normal"
    ball[i], first_target[i] = ball_pos(ball[i], first_target[i])
    ball[i].score = 0
    ball[i].target_condition = "not_hit"

# Test for initial collision from initialising balls.
# Whilst true each balls position will be changed.
while True:
    number_collision = 0
    for fball in range(len(ball)):
        for sball in range(len(ball)):
            #Calculate distance between all objects
            distance_between = dist_calculator(ball[fball].pos, ball[sball].pos)
            if sball != fball:
                #If colliding
                if distance_between <= 2 * radius_ball:
                    #Redefine ball position
                    ball[sball], first_target[sball] = ball_pos(ball[sball], first_target[sball])
                    number_collision += 1
    #If not objects are colliding break
    if number_collision == 0:
        break

target = sphere(color=color.black, radius=radius_target, make_trail=True, retain=200)
target.p = vector(0, 0, 0)

dt = 0.01
dy = 0.01
dx = 0.1
reruns = 10
mean_score = 0

# plt.plot(lower_limit, weighting)
# plt.legend()
# plt.savefig("pic{}.png".format(rerun))

#Initilise empty list
empty = []
score_list = []
#Initlise empty score list
score_list = [[] for _ in range(len(ball))]
#Initialise weighting
weighting = [1 / len(lower_limit)] * len(lower_limit)
#Initialise temp weighting
weighting_temp = [1 / len(lower_limit)] * len(lower_limit)
#Initiliase empty mean reward list
mean_reward_list = [[] for _ in range(len(lower_limit))]
#Initliase mean score list
mean_score_list = []
#Run across all defined runs
for rerun in range(reruns):
    rate(1000)
    choices = [-1, 1]
    #Set current score to zero
    current_score = 0
    #Equivalent of predicting random score for e greedy
    predict_proba = random.randint(1, 11) * 0.1
    # Equivalent of 80% probability
    if predict_proba >= 0.3:
        # Find max weighting
        max_weight = max(weighting)
        # Find index of max value
        item = weighting.index(max_weight)

    #If probability is less than 80%
    elif predict_proba < 0.3:
        item_choice = random.choice(lower_limit)
        item = lower_limit.index(item_choice)
    
    item = int(item)
    # Setting the item as definite
    item = 0
    if rerun == 0:
        for temp_limit in range(len(lower_limit)):
            item = int(temp_limit)
            print(item)
            current_score = run_program(item, current_score)
            weighting, mean_score, current_score, mean_score_list, mean_reward_list = calclate_weighting(item, weighting, mean_score, current_score, mean_score_list, mean_reward_list, rerun, mode=1)
    elif rerun != 0:
        current_score = run_program(item, current_score)
        weighting, mean_score, current_score, mean_score_list, mean_reward_list = calclate_weighting(item, weighting, mean_score, current_score, mean_score_list, mean_reward_list, rerun, mode=1)
        plt.plot(lower_limit, weighting)

    for in_score in range(len(ball)):
        #If rerun is zero set the score to zero
        if rerun == 0:
            ball[in_score].score = 0
        elif rerun != 0:
            score_ball = ball[in_score].score
            #Add ball score to list
            score_list[in_score].append(score_ball)
            #set the score to zero
            ball[in_score].score = 0
    print("score_list", score_list)
# plt.show()
plt.savefig("weighting_100{}.png".format(number_itterations))
plt.close()

mean_score_list = []
standard_eror_list = []

for itter in range(len(score_list)):
    mean_score = sum(score_list[itter]) / len(score_list[itter])
    mean_score_list.append(mean_score)
    std = sum([(item - mean_score) ** 2 for item in score_list[itter]]) / (len(score_list[itter])) ** 0.5
    st_error = std / (len(score_list[itter]) ** 0.5)
    standard_eror_list.append(st_error)
    print(itter, mean_score)
# plt.plot(range(len(ball)), score_list)
# plt.plot(range(len(ball)), mean_score_list, xerr = standard_eror_list)
plt.errorbar(range(len(ball)), mean_score_list, yerr=standard_eror_list, fmt='o')

plt.xlabel("Number of objects")

plt.ylabel("Object score")
plt.ylim(bottom=0)
# plt.show()
plt.savefig("score_50_{}.png".format(number_itterations))
plt.close()

