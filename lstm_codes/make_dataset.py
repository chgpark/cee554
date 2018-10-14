import random
import math
import csv

ISZIGZAG = True
UNCERTAINTY = 0.03
DIMENSION = '3D'
DELTALENGTH = 0.05
ONESIDELENGTH = 4

class position:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


class UWB(position):
    def __init__(self, x, y, z):
        super(UWB, self).__init__(x, y, z)

    def getDistance(self,robot):
        distance = math.sqrt((self.x -robot.x)**2 + (self.y - robot.y)**2 + (self.z -robot.z)**2)
        return distance

    def getDistancewNoise(self, robot):
        distance = self.getDistance(robot)
        distance = distance*(1.0 + random.random()*UNCERTAINTY)

        return distance

class Robot(position):
    def __init__(self, x, y, z):
        super(Robot, self).__init__(x, y, z)

    def getPose(self):
        return (self.x, self.y, self.z)

    def setPose(self,dx,dy,dz):
        self.x += dx
        self.y += dy
        self.z += dz


def curve_function1(x):
    return -math.sqrt(25 - x**2) + 5
def curve_function2(x):
    return math.sqrt((10 - x) * x)

class CSVWriter():
    def __init__(self, wr, kobuki):
        self.dimension = DIMENSION
        self.wr = wr
        self.kobuki = kobuki
        self.iteration_num = int(ONESIDELENGTH/DELTALENGTH)
    def writerow(self, dist_list):
        if (self.dimension == '2D'):
            self.wr.writerow(dist_list +(self.kobuki.x, self.kobuki.y))

        elif (self.dimension == '3D'):
            written_data = dist_list + (self.kobuki.x, self.kobuki.y, self.kobuki.z)
            for uwb in uwb_list:
                written_data += (self.kobuki.x - uwb.x, self.kobuki.y - uwb.y, self.kobuki.z - uwb.z)
            self.wr.writerow(written_data)

    def moveRobot(self, x,y,z):
        self.kobuki.setPose(x, y, z)
        dist1 = uwb1.getDistancewNoise(kobuki)
        dist2 = uwb2.getDistancewNoise(kobuki)
        dist3 = uwb3.getDistancewNoise(kobuki)
        dist4 = uwb4.getDistancewNoise(kobuki)
        return dist1,dist2,dist3,dist4

    def zigzag_xy(self, round_number):
        for j in range(5):
            for i in range(self.iteration_num):
                dist_list = self.moveRobot(DELTALENGTH, 0.0, 0.0)
                self.writerow(dist_list)
            for i in range(int(self.iteration_num/10)):
                dist_list = self.moveRobot(0.0, DELTALENGTH, 0.0)
                self.writerow(dist_list)
            for i in range(int(self.iteration_num)):
                dist_list = self.moveRobot(-DELTALENGTH, 0.0, 0.0)
                self.writerow(dist_list)
            for i in range(int(self.iteration_num/10)):
                dist_list = self.moveRobot(0.0, DELTALENGTH, 0.0)
                self.writerow(dist_list)

        for i in range(self.iteration_num):
            dist_list = self.moveRobot(DELTALENGTH, 0.0, 0.0)
            self.writerow(dist_list)

        for j in range(5):
            for i in range(self.iteration_num):
                dist_list = self.moveRobot(0.0, -DELTALENGTH, 0.0)
                self.writerow(dist_list)
            for i in range(int(self.iteration_num / 10)):
                dist_list = self.moveRobot(-DELTALENGTH, 0.0, 0.0)
                self.writerow(dist_list)
            for i in range(self.iteration_num):
                dist_list = self.moveRobot( 0.0,DELTALENGTH, 0.0)
                self.writerow(dist_list)
            for i in range(int(self.iteration_num / 10)):
                dist_list = self.moveRobot(-DELTALENGTH, 0.0,  0.0)
                self.writerow(dist_list)

        for i in range(self.iteration_num):
            dist_list = self.moveRobot(0.0, -DELTALENGTH, 0.0)
            self.writerow(dist_list)

    def zigzag_yz(self, round_number):
        for j in range(5):
            for i in range(self.iteration_num):
                dist_list = self.moveRobot(0.0, DELTALENGTH , 0.0)
                self.writerow(dist_list)
            for i in range(int(self.iteration_num/10)):
                dist_list = self.moveRobot(0.0, 0.0, -DELTALENGTH)
                self.writerow(dist_list)
            for i in range(int(self.iteration_num)):

                dist_list = self.moveRobot(0.0, -DELTALENGTH , 0.0)
                self.writerow(dist_list)
            for i in range(int(self.iteration_num/10)):
                dist_list = self.moveRobot(0.0, 0.0, -DELTALENGTH)
                self.writerow(dist_list)

        for i in range(self.iteration_num):
            dist_list = self.moveRobot(0.0, DELTALENGTH, 0.0)
            self.writerow(dist_list)

        for j in range(5):
            for i in range(self.iteration_num):
                dist_list = self.moveRobot(0.0, 0.0, DELTALENGTH)
                self.writerow(dist_list)
            for i in range(int(self.iteration_num / 10)):
                dist_list = self.moveRobot(0.0, -DELTALENGTH, 0.0)
                self.writerow(dist_list)
            for i in range(self.iteration_num):
                dist_list = self.moveRobot(0.0, 0.0, -DELTALENGTH)
                self.writerow(dist_list)
            for i in range(int(self.iteration_num / 10)):
                dist_list = self.moveRobot(0.0, -DELTALENGTH, 0.0)
                self.writerow(dist_list)

        for i in range(self.iteration_num):
            dist_list = self.moveRobot(0.0, 0.0, DELTALENGTH)
            self.writerow(dist_list)

    def zigzag_zx(self, round_number):
        for j in range(5):
            for i in range(self.iteration_num):
                dist_list = self.moveRobot( 0.0, 0.0, -DELTALENGTH)
                self.writerow(dist_list)
            for i in range(int(self.iteration_num/10)):
                dist_list = self.moveRobot( -DELTALENGTH, 0.0, 0.0)
                self.writerow(dist_list)
            for i in range(int(self.iteration_num)):
                dist_list = self.moveRobot(0.0, 0.0, DELTALENGTH)
                self.writerow(dist_list)
            for i in range(int(self.iteration_num/10)):
                dist_list = self.moveRobot( -DELTALENGTH,  0.0, 0.0)
                self.writerow(dist_list)

        for i in range(self.iteration_num):
            dist_list = self.moveRobot( 0.0, 0.0, -DELTALENGTH)
            self.writerow(dist_list)

        for j in range(5):
            for i in range(self.iteration_num):
                dist_list = self.moveRobot(DELTALENGTH, 0.0, 0.0)
                self.writerow(dist_list)
            for i in range(int(self.iteration_num / 10)):
                dist_list = self.moveRobot(0.0, 0.0, DELTALENGTH)
                self.writerow(dist_list)
            for i in range(self.iteration_num):
                dist_list = self.moveRobot(-DELTALENGTH, 0.0, 0.0)
                self.writerow(dist_list)
            for i in range(int(self.iteration_num / 10)):
                dist_list = self.moveRobot(0.0, 0.0, DELTALENGTH)
                self.writerow(dist_list)

        for i in range(self.iteration_num):
            dist_list = self.moveRobot(DELTALENGTH, 0.0, 0.0)
            self.writerow(dist_list)

    def drawZigzagPath_3D(self, round_number):
        self.zigzag_xy(round_number)
        for j in range(10):
            for k in range(int(self.iteration_num/10)):
                dist_list = self.moveRobot( 0.0, 0.0, DELTALENGTH)
                self.writerow(dist_list)
            self.zigzag_xy(round_number)

        self.zigzag_yz(round_number)
        for j in range(10):
            for k in range(int(self.iteration_num/10)):
                dist_list = self.moveRobot( DELTALENGTH, 0.0, 0.0)
                self.writerow(dist_list)
            self.zigzag_yz(round_number)

        self.zigzag_zx(round_number)
        for j in range(10):
            for k in range(int(self.iteration_num/10)):
                dist_list = self.moveRobot( 0.0, DELTALENGTH, 0.0)
                self.writerow(dist_list)
            self.zigzag_zx(round_number)

    def drawSquarePath(self, round_number):
        for j in range(round_number):
            for i in range(self.iteration_num):
                dist_list = self.moveRobot(DELTALENGTH, 0.0, 0.0)
                self.writerow(dist_list)
            for i in range(self.iteration_num):
                dist_list = self.moveRobot(0.0, DELTALENGTH, 0.0)
                self.writerow(dist_list)
            for i in range(self.iteration_num):
                dist_list = self.moveRobot(-DELTALENGTH, 0.0, 0.0)
                self.writerow(dist_list)
            for i in range(self.iteration_num):
                dist_list = self.moveRobot(0.0, -DELTALENGTH, 0.0)
                self.writerow(dist_list)


    def drawDiagonalPath(self):
        for i in range(self.iteration_num):
            dist_list = self.moveRobot(DELTALENGTH, DELTALENGTH, 0.0)
            self.writerow(dist_list)

    def drawTestPath(self):
        # 0,0 -> 1,0
        for i in range(100):
            dist_list = self.moveRobot(DELTALENGTH, 0.0, 0.0)
            self.writerow(dist_list)
        # -> 2,1.5
        for i in range(200):
            dist_list = self.moveRobot(DELTALENGTH / 2, DELTALENGTH * 3 / 4, 0.0)
            self.writerow(dist_list)
        # -> 4,0
        for i in range(250):
            dist_list = self.moveRobot(DELTALENGTH * 4 / 5, -DELTALENGTH * 3 / 5, 0.0)
            self.writerow(dist_list)
        # -> 5,4
        for i in range(412):
            dist_list = self.moveRobot(DELTALENGTH / 4.12, DELTALENGTH * 4 / 4.12, 0.0)
            self.writerow(dist_list)
        # -> 1,5
        for i in range(412):
            dist_list = self.moveRobot(-DELTALENGTH * 4 / 4.12, DELTALENGTH / 4.12, 0.0)
            self.writerow(dist_list)
        # -> 4,4
        for i in range(315):
            dist_list = self.moveRobot(DELTALENGTH * 3 / 3.15, -DELTALENGTH / 3.15, 0.0)
            self.writerow(dist_list)
        # -> 3,3
        for i in range(100):
            dist_list = self.moveRobot(-DELTALENGTH, -DELTALENGTH, 0.0)
            self.writerow(dist_list)
        # -> 2,4
        for i in range(100):
            dist_list = self.moveRobot(-DELTALENGTH, DELTALENGTH, 0.0)
            self.writerow(dist_list)
        # -> 1, 1.5
        for i in range(269):
            dist_list = self.moveRobot(-DELTALENGTH * 2 / 5.38, -DELTALENGTH * 5 / 5.38, 0.0)
            self.writerow(dist_list)

        # -> 1.75, 2
        for i in range(90):
            dist_list = self.moveRobot(DELTALENGTH * 3 / 3.6, DELTALENGTH * 2 / 3.6, 0.0)
            self.writerow(dist_list)

        # -> 1.25, 2.75
        for i in range(90):
            dist_list = self.moveRobot(-DELTALENGTH * 2 / 3.6, DELTALENGTH * 3 / 3.6, 0.0)
            self.writerow(dist_list)

        # 1.25, 1.95
        for i in range(80):
            dist_list = self.moveRobot(0.0, -DELTALENGTH, 0.0)
            self.writerow(dist_list)
        # 0, 0
        print(self.kobuki.x, self.kobuki.y)
        for i in range(231):
            dist_list = self.moveRobot(-DELTALENGTH * 1.25 / 2.31, -DELTALENGTH * 1.95 / 2.31, 0.0)
            self.writerow(dist_list)
        print(self.kobuki.x, self.kobuki.y)





uwb1 = UWB( -2.4, -2.4, 0.1)
uwb2 = UWB( 2.4, -2.4, 2.7)
uwb3 = UWB( 2.4, 2.4, 1.4)
uwb4 = UWB( -2.4, 2.4, 4.2)

uwb_list = [uwb1, uwb2, uwb3, uwb4]
# Below :
# uwb1 = UWB( 0.9, 0.9, 0)
# uwb2 = UWB( 4.5,-0.9, 0)
# uwb3 = UWB( 4.5, 4.5, 0)
# uwb4 = UWB( 0.9, 2.7, 0)
file_name = 'train_' + DIMENSION
# file_name = 'test_data_arbitrary_path' + DIMENSION
if (ISZIGZAG):
    file_name = file_name +'_' + 'zigzag'
file_name = file_name + '.csv'

kobuki = Robot(-2.0, -2.0, 4.0)

train_file = open(file_name ,'w',encoding = 'utf-8', newline ='')
wr = csv.writer(train_file)

dataWriter = CSVWriter(wr, kobuki)

dataWriter.drawZigzagPath_3D(10)


print ("Make "+file_name)


train_file.close()
