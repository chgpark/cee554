import random
import math
import csv

ISZIGZAG = False
UNCERTAINTY = 0.03
DIMENSION = '3D'
DELTALENGTH = 0.05
ONESIDELENGTH = 4

class position(object):
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

    def setPosition(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def setPose(self,dx,dy,dz):
        self.x += dx
        self.y += dy
        self.z += dz


def curve_function1(x):
    return -math.sqrt(25 - x**2) + 5
def curve_function2(x):
    return math.sqrt((10 - x) * x)

class CSVWriter():
    def __init__(self, wr, drone):
        self.dimension = DIMENSION
        self.wr = wr
        self.drone = drone
        self.iteration_num = int(ONESIDELENGTH/DELTALENGTH)
    def writerow(self, dist_list):
        if (self.dimension == '2D'):
            self.wr.writerow(dist_list +(self.drone.x, self.drone.y))

        elif (self.dimension == '3D'):
            written_data = dist_list + (self.drone.x, self.drone.y, self.drone.z)
            for uwb in uwb_list:
                written_data += (self.drone.x - uwb.x, self.drone.y - uwb.y, self.drone.z - uwb.z)
            self.wr.writerow(written_data)

    def moveRobot(self, x,y,z):
        self.drone.setPose(x, y, z)
        dist1 = uwb1.getDistancewNoise(drone)
        dist2 = uwb2.getDistancewNoise(drone)
        dist3 = uwb3.getDistancewNoise(drone)
        dist4 = uwb4.getDistancewNoise(drone)
        return dist1,dist2,dist3,dist4

    def zigzag_xy(self):
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

    def zigzag_yz(self):
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

    def zigzag_zx(self):
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
        for i in range(round_number):
            self.drone.setPosition(-2, -2, 0)
            self.zigzag_xy()
            for j in range(10):
                for k in range(int(self.iteration_num/10)):
                    dist_list = self.moveRobot(0.0, 0.0, DELTALENGTH)
                    self.writerow(dist_list)
                self.zigzag_xy()

            self.zigzag_yz()
            for j in range(10):
                for k in range(int(self.iteration_num/10)):
                    dist_list = self.moveRobot(DELTALENGTH, 0.0, 0.0)
                    self.writerow(dist_list)
                self.zigzag_yz()

            self.zigzag_zx()
            for j in range(10):
                for k in range(int(self.iteration_num/10)):
                    dist_list = self.moveRobot(0.0, DELTALENGTH, 0.0)
                    self.writerow(dist_list)
                self.zigzag_zx()

            for k in range(139):
                dist_list = self.moveRobot(-DELTALENGTH/1.7320508, -DELTALENGTH/1.7320508, -DELTALENGTH/1.7320508)
                self.writerow(dist_list)

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
        print(self.drone.x, self.drone.y)
        for i in range(231):
            dist_list = self.moveRobot(-DELTALENGTH * 1.25 / 2.31, -DELTALENGTH * 1.95 / 2.31, 0.0)
            self.writerow(dist_list)
        print(self.drone.x, self.drone.y)

    def drawPolyPath(self):
        # 0,0 -> (1.5,0,1.5)
        for i in range(30):
            dist_list = self.moveRobot(DELTALENGTH, 0.0, (0.667*(self.drone.x + DELTALENGTH) ** 2) - self.drone.z)
            self.writerow(dist_list)
        print(self.drone.x, self.drone.y, self.drone.z)
            # (1.5,0,1.5) -> (0,0,3)

        for i in range(30):
            dist_list = self.moveRobot(- DELTALENGTH, 0.0, (0.667 * (self.drone.x - DELTALENGTH -1.5) ** 2 +1.5) - self.drone.z)
            self.writerow(dist_list)
        print(self.drone.x, self.drone.y, self.drone.z)

        #draw a square
        for i in range(30):
            dist_list = self.moveRobot(0,DELTALENGTH,0)
            self.writerow(dist_list)
        print(self.drone.x, self.drone.y, self.drone.z)

        for i in range(30):
            dist_list = self.moveRobot(-DELTALENGTH,0,0)
            self.writerow(dist_list)
        print(self.drone.x, self.drone.y, self.drone.z)

        for i in range(60):
            dist_list = self.moveRobot(0,-DELTALENGTH,0)
            self.writerow(dist_list)
        print(self.drone.x, self.drone.y, self.drone.z)

        for i in range(60):
            dist_list = self.moveRobot(+DELTALENGTH,0,0)
            self.writerow(dist_list)
        print(self.drone.x, self.drone.y, self.drone.z)

        for i in range(60):
            dist_list = self.moveRobot(0,+DELTALENGTH,0)
            self.writerow(dist_list)
        print(self.drone.x, self.drone.y, self.drone.z)

        for i in range(30):
            dist_list = self.moveRobot(-DELTALENGTH,0,0)
            self.writerow(dist_list)
        print(self.drone.x, self.drone.y, self.drone.z)


    def drawSpiralPath(self):
        # (0,0,0)=>(1.2,0,0.7) z=0.486*x^2
        for i in range(24):
            dist_list = self.moveRobot(DELTALENGTH, 0.0, (0.486*(self.drone.x+DELTALENGTH)**2 )- self.drone.z)
            self.writerow(dist_list)
        print(self.drone.x, self.drone.y, self.drone.z)

        #spiral 48*0.05 =2.4 / radius = 1.2  # CCW
        for i in range(48):
            dist_list = self.moveRobot(-DELTALENGTH, math.sqrt((1.2*1.2+0.001) - (self.drone.x -DELTALENGTH)**2) -self.drone.y, DELTALENGTH*(0.2))
            self.writerow(dist_list)
        print(self.drone.x, self.drone.y, self.drone.z)

        for i in range(48):
            #print(math.sqrt((1.2 * 1.2 + 0.001) - (self.drone.x + DELTALENGTH) ** 2))
            dist_list = self.moveRobot(+DELTALENGTH, -1*math.sqrt((1.2*1.2+0.001) - (self.drone.x +DELTALENGTH)**2)-self.drone.y, DELTALENGTH*(0.2))
            self.writerow(dist_list)
        print(self.drone.x, self.drone.y, self.drone.z)

        for i in range(48):
            dist_list = self.moveRobot(-DELTALENGTH, math.sqrt((1.2 * 1.2 + 0.001) - (self.drone.x - DELTALENGTH)**2) - self.drone.y, DELTALENGTH * (0.2))
            self.writerow(dist_list)
        print(self.drone.x, self.drone.y, self.drone.z)

        for i in range(48):
            dist_list = self.moveRobot(+DELTALENGTH, -1 * math.sqrt((1.2 * 1.2 + 0.001) - (self.drone.x + DELTALENGTH)**2) - self.drone.y, DELTALENGTH * (0.2))
            self.writerow(dist_list)
        print(self.drone.x, self.drone.y, self.drone.z)

        for i in range(38):
            dist_list = self.moveRobot(-DELTALENGTH, math.sqrt((1.2 * 1.2 + 0.001) - (self.drone.x - DELTALENGTH)**2) - self.drone.y, DELTALENGTH * (0.2))
            self.writerow(dist_list)
        print(self.drone.x, self.drone.y, self.drone.z)



'''Should be fixed!!'''
uwb1 = UWB( -2.4, -2.4, 0.1)
uwb2 = UWB( 2.4, -2.4, 2.7)
uwb3 = UWB( 2.4, 2.4, 1.4)
uwb4 = UWB( -2.4, 2.4, 4.2)


#uwb_list = [uwb1, uwb2, uwb3, uwb4]
# Below :
# uwb1 = UWB( 0.9, 0.9, 0)
# uwb2 = UWB( 4.5,-0.9, 0)
# uwb3 = UWB( 4.5, 4.5, 0)
# uwb4 = UWB( 0.9, 2.7, 0)

uwb_list = [uwb1, uwb2, uwb3, uwb4]

file_name = 'spiral_' + DIMENSION # file name
# file_name = 'test_data_arbitrary_path' + DIMENSION
if (ISZIGZAG):
    file_name = file_name +'_' + 'zigzag_200'
file_name = file_name + '.csv'

drone = Robot(0.0, 0.0, 0.0) #initial position


#train_file = open(file_name,'w',encoding = 'utf-8', newline ='')
train_file = open(file_name,'w')
wr = csv.writer(train_file)

dataWriter = CSVWriter(wr, drone)

# dataWriter.drawZigzagPath_3D(30)
#dataWriter.drawTestPath()
# dataWriter.drawPolyPath()
dataWriter.drawSpiralPath()

print ("Make "+file_name)


train_file.close()
