#!/usr/bin/env python
from math import *
import random
from environment_settings import UWB_LOC
from environment_settings import MAP_X, MAP_Y, MAP_Z
# UWB_LOC = [[0.90, 0.90, 0 ], [4.5 , -0.90, 0],
#            [4.50, 4.50, 0 ], [0.90,   2.7, 0]] # temporary location input
DISCONNECTED = -1
SAMPLING_NUM = 1000
UNCERTAINTY = 0.095
SENSOR_NOISE = 0.002
TOP_PARTICLE_RATIO = 10 # for estimate position via particles.

class Position(object):
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
    def setPosition(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

class UWBAnchor(Position):
    def __init__(self, position_list):
        super(UWBAnchor, self).__init__(position_list[0], position_list[1], position_list[2])
        self.isLOS = True
        self.standard_deviation = 0.

    def setRange(self,range):
        self.range = range

class Particle(Position):
    def __init__(self):
        x = random.random() * MAP_X
        y = random.random() * MAP_Y
        z = random.random() * MAP_Z

        super(Particle, self).__init__(x, y, z)

        self.sensor_noise = SENSOR_NOISE

    def setHeight(self, measured_z):
        self.z = measured_z + self.sensor_noise*random.random()

    def scatterPosition(self, particle):
        while True:
            x = particle.x + (random.random() - 0.5) * UNCERTAINTY
            y = particle.y + (random.random() - 0.5) * UNCERTAINTY
            z = particle.z + (random.random() - 0.5) * UNCERTAINTY

            if z < 0:
                z = (-1)*z

            if ((x - particle.x)**2 + (y - particle.y)**2 + (z - particle.z)**2) <= 0.25*UNCERTAINTY:
                break

        super(Particle, self).setPosition(x, y, z)

    def scatterPositionWithLiDAR(self, particle, measured_z):
        while True:
            x = particle.x + (random.random() - 0.5) * UNCERTAINTY
            y = particle.y + (random.random() - 0.5) * UNCERTAINTY

            z = measured_z + self.sensor_noise*random.random()

            if ((x - particle.x)**2 + (y - particle.y)**2) <= 0.25 * UNCERTAINTY:
                break

        super(Particle, self).setPosition(x, y, z)

class MonteCarloLocalization(object):
    def __init__(self):
        self.samples_num = SAMPLING_NUM
        # self.resampling_num = RESAMPLING_NUM
        self.initialize()
        self.weights_list = [1/SAMPLING_NUM]*SAMPLING_NUM

    def initialize(self):
        self.particles = []
        for i in range(self.samples_num):
            p = Particle()
            self.particles.append(p)

    def getEuclidianDistance(self, Particle, UWB):
        return sqrt((Particle.x - UWB.x)**2 + (Particle.y - UWB.y)**2 + (Particle.z - UWB.z)**2)

    def calculateDistanceDiff(self, uwb_list):
        z = []
        for particle in self.particles:
            error = 0
            for uwb in uwb_list:
                if uwb.isLOS:
                    error += (self.getEuclidianDistance(particle, uwb) - uwb.range)**2
            z.append(error)
        return z

    def calculateDistanceDiffWithLiDAR(self, uwb_list, LiDAR_data, LiDAR_data_weight):
        z = []
        for particle in self.particles:
            error = 0
            for uwb in uwb_list:
                if uwb.isLOS:
                    error += (self.getEuclidianDistance(particle, uwb) - uwb.range)**2

            error += LiDAR_data_weight*(particle.z - LiDAR_data)**2

            z.append(error)
        return z
    def getGaussianDistribution(self, mu_minus_x, sigma):
        return exp(- ((mu_minus_x) ** 2) / (sigma ** 2) / 2.0) / sqrt(2.0 * pi * (sigma ** 2))

    def getMultivariateGaussianDistribution(self,mu_minus_x_list, std_list):
        epsilon = 0.000000001
        assert len(mu_minus_x_list) == len(std_list)
        dimension = len(mu_minus_x_list)
        det_std = 1
        for std in std_list:
            det_std = det_std*std
        denominator = pow(2.0*pi, dimension/2.0) * sqrt(det_std)
        exp_power = 0
        for i in range(dimension):
            exp_power += (mu_minus_x_list[i]/(std_list[i] + epsilon))**2
        exp_power = exp_power*(- 0.5)
        weight = exp(exp_power)/(denominator + epsilon)
        return weight

    def setWeightsbyGaussian(self, uwb_list, LiDAR_data, LiDAR_data_weight):
        self.weights_list = []
        for particle in self.particles:
            UWB_mu_minus_x_list = []
            UWB_std_list = []
            weight = 0
            for uwb in uwb_list:
                if uwb.isLOS:
                    mu_minus_x = self.getEuclidianDistance(particle, uwb)
                    UWB_mu_minus_x_list.append(mu_minus_x)
                    UWB_std_list.append(uwb.standard_deviation)

                    weight = self.getMultivariateGaussianDistribution(UWB_mu_minus_x_list, UWB_std_list)
            LiDAR_weight = self.getGaussianDistribution(particle.z - LiDAR_data, 0.04)
            weight += LiDAR_weight * LiDAR_data_weight
            self.weights_list.append(weight)

        self.weights_list = self.normalizeList(self.weights_list)

    def calculateDistanceDiffConsideringMultiPath(self, uwb_list):
        z = []
        for particle in self.particles:
            error = 0
            for uwb in uwb_list:
                self.checker.setParticleAnchor(particle, uwb)
                if uwb.isLOS:
                    if self.checker.isObstaclesBlockBTWParticleAndAnchor():
                        if self.checker.getEdgeValue(particle, uwb) == DISCONNECTED:
                            self.checker.graph.resetAllStates()
                            self.checker.setGraphEdges()
                            self.checker.graph.doDijkstra(0)

                            base = self.checker.graph.distance[-1]
                            height = particle.z - uwb.z
                            expected_z_value = self.checker.calculatebyPythagoreanTheorem(base, height)
                        else:
                            expected_z_value = self.getEuclidianDistance(particle, uwb)

                    else:
                        expected_z_value = self.getEuclidianDistance(particle, uwb)

                    try:
                        error += (expected_z_value - uwb.range)**2

                    except UnboundLocalError:
                        print (self.checker.isObstaclesBlockBTWParticleAndAnchor())
                        print (self.checker.getEdgeValue(particle, uwb))
                        print (particle.x, particle.y, particle.z, uwb.x, uwb.y, uwb.z)

            z.append(error)
        return z

    def calculateCosineSimilarity(self, uwb_list):
        z = []
        dot_product_sum = 0
        absolute_val_d = 0
        absolute_val_d_hat = 0
        for particle in self.particles:
            similarity = 0
            for uwb in uwb_list:
                if uwb.isLOS:
                    dot_product_sum += uwb.range*self.getEuclidianDistance(particle, uwb)
                    absolute_val_d += uwb.range**2
                    absolute_val_d_hat += self.getEuclidianDistance(particle, uwb)**2
            #  K(X, Y) = <X, Y> / (||X||*||Y||)
            similarity = dot_product_sum/(sqrt(absolute_val_d)*sqrt(absolute_val_d_hat))
            z.append(similarity)
        return z

    def normalizeList(self, weights_list):
        denominator = sum(self.weights_list)
        return list(map(lambda x: x/denominator, self.weights_list))


    def setWeights(self,z):
        for i, z_i in enumerate(z):
            y = (1/(z_i+ 0.0000001)) #*self.weights_list[i]
            self.weights_list[i] = y

        self.weights_list = self.normalizeList(self.weights_list)

    def resampling(self):
        probs = [sum(self.weights_list[:i+1]) for i in range(self.samples_num)]
        weights_list = []
        particles_list = []
        for n in range(self.samples_num):
            random_value = random.random()
            for i in range(self.samples_num):
                if random_value <= probs[i]:
                    particles_list.append(self.particles[i])
                    weights_list.append(self.weights_list[i])
                    break

        return particles_list, weights_list

    def scatterParticle(self, particles_list):
        for i, selected_particle in enumerate(particles_list):
            self.particles[i].scatterPosition(selected_particle)

    def scatterParticleWithLiDAR(self, particles_list, measured_LiDAR_data):
        for i, selected_particle in enumerate(particles_list):
            self.particles[i].scatterPositionWithLiDAR(selected_particle, measured_LiDAR_data)

    def getEstimate(self, particles_list, weights_list, method = 'top10%'):

        estimated_position = Position(0, 0, 0)

        if (method == 'max'):
            idx = weights_list.index(max(weights_list))
            estimated_position.x = particles_list[idx].x
            estimated_position.y = particles_list[idx].y
            estimated_position.z = particles_list[idx].z

        elif (method =='all'):
            for i, particle in enumerate(particles_list):
                weights_list = self.normalizeList(weights_list)
                estimated_position.x += weights_list[i]*particle.x
                estimated_position.y += weights_list[i]*particle.y
                estimated_position.z += weights_list[i]*particle.z

        elif (method =='top10%'):
            particle_weight_tuple_list = []
            for i in range(len(weights_list)):
                particle_weight_tuple_list.append((particles_list[i], weights_list[i]))

            particle_weight_tuple_list.sort(key = lambda element:element[1], reverse = True)

            upper10_list = particle_weight_tuple_list[:int(self.samples_num/TOP_PARTICLE_RATIO)]

            #   normalize upper 10 percent weights
            denominator = 0
            for particle, weight in upper10_list:
                denominator += weight

            for particle,weight in upper10_list:
                norm_weight = weight/denominator
                estimated_position.x += norm_weight*particle.x
                estimated_position.y += norm_weight*particle.y
                estimated_position.z += norm_weight*particle.z

        return estimated_position

    def doMCL(self, uwb_list):
        z = self.calculateDistanceDiff(uwb_list)
        weights = self.setWeights(z)
        particles_list = self.resampling()
        # xyz_list = self.printMaxParticle(particles_list)
        self.scatterParticle(particles_list)

        # return xyz_list

def calculateMSE(pd_line, xyz_list):
    MSE_partial = 0
    for i in range(3):
        MSE_partial += (pd_line[i] - xyz_list[i])**2
    return MSE_partial


