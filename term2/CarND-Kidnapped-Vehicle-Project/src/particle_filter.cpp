/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;
default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1.
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

    num_particles = 100;

    weights.resize(num_particles);
    particles.resize(num_particles);

    cout << "ParticleFilter::init called with " << num_particles << " number of particles. \n";

    double std_x, std_y, std_theta;

    std_x = std[0];
    std_y = std[1];
    std_theta = std[2];

    normal_distribution<double> dist_x(x, std_x);
    normal_distribution<double> dist_y(y, std_y);
    normal_distribution<double> dist_theta(theta, std_theta);


    for (int i = 0; i < num_particles; i++){
        Particle temp_particle;
        temp_particle.id = i;
        temp_particle.x = dist_x(gen);
        temp_particle.y = dist_y(gen);
        temp_particle.theta = dist_theta(gen);
        temp_particle.weight = 1.0;

        particles[i] = temp_particle;

        cout << i << " : " << temp_particle.x << "\t" << temp_particle.y << "\t" << temp_particle.theta << endl;
    }
    is_initialized = true;
    cout << "Number of particles initialised is " << particles.size() << endl;
    cout << " ParticleFilter::init - call over \n";
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	cout << "ParticleFilter::prediction called \n" ;
	//cout << "ParticleFilter::prediction - delta_t=" << delta_t << ", velocity=" << velocity << ", yaw_rate=" << yaw_rate << "\n" ;

    normal_distribution<double> dist_x(0, std_pos[0]);
    normal_distribution<double> dist_y(0, std_pos[1]);
    normal_distribution<double> dist_theta(0, std_pos[2]);

    for (int i = 0; i < num_particles; i++){
        if (fabs(yaw_rate) > 0.0001){ // YAW_RATE is non-zero
            particles[i].x = particles[i].x + velocity/yaw_rate * (sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta));
            particles[i].y = particles[i].y + velocity/yaw_rate * (-cos(particles[i].theta + yaw_rate * delta_t) + cos(particles[i].theta));
            particles[i].theta = particles[i].theta + yaw_rate * delta_t;
        }
        else{ // YAW_RATE is zero
            particles[i].x = particles[i].x + velocity * cos(particles[i].theta);
            particles[i].y = particles[i].y + velocity * sin(particles[i].theta);
        }
        particles[i].x = particles[i].x + dist_x(gen);
        particles[i].y = particles[i].y + dist_y(gen);
        particles[i].theta = particles[i].theta + dist_theta(gen);
    }
	cout << "ParticleFilter::prediction - call over \n" ;
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
	//   implement this method and use it as a helper during the updateWeights phase.

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

    // For every particle
    for (int i = 0; i < num_particles; i++){

        double particle_id    = particles[i].id;
        double particle_x_pos = particles[i].x;
        double particle_y_pos = particles[i].y;
        double particle_theta = particles[i].theta;
        //double particle_weight= particles[i].weight;
        //particles[i].associations.clear();

        // Create a vector of size of map landmarks that holds the distance
        // within the sensor range of the particle
        int num_landmarks = map_landmarks.landmark_list.size();
        vector<int> landmark_2_particle_sensed;

        for (int k = 0 ; k < num_landmarks ; k++){
            float map_x = map_landmarks.landmark_list[k].x_f;
            float map_y = map_landmarks.landmark_list[k].y_f;
            // if the Euclidean distance between the particle and
            // map landmark is within sensor range
            if (dist(particle_x_pos, particle_y_pos, map_x, map_y) <= 50){
                // Convert observation  from vehicle co-ordinate to particle/map co-ordinate
                landmark_2_particle_sensed.push_back(k+1); // +1 is because the landmarks in map starts with 1
                //cout<< "Map location within range is : " << k << endl;
            }
        }
        cout << "No. of sensed objects within range are : " << landmark_2_particle_sensed.size() << endl;
        for ( int m = 0 ; m < landmark_2_particle_sensed.size(); m++){
            cout << landmark_2_particle_sensed[m] << " ";
        }
        cout << endl;
        // For Every Observation check the nearest map landmark location
        // obs_2_map_nn will hold the nearest neighbour to that observation
        vector<double> obs_2_map_nn;
        double multiVarGD = 1.0;
        for (int j = 0; j < observations.size() ; j++){
            // comvert from vehicle co-ordinate to map co-ordinate
            double obs_2_map_x = observations[j].x * cos(particle_theta) - observations[j].y *  sin(particle_theta) + particle_x_pos;
            double obs_2_map_y = observations[j].x * sin(particle_theta) + observations[j].y *  cos(particle_theta) + particle_y_pos;

            vector<double> obs_2_map_distance;
            for (int k = 0; k < landmark_2_particle_sensed.size(); k++){
                int map_id = landmark_2_particle_sensed[k];
                float map_x = map_landmarks.landmark_list[map_id - 1].x_f;
                float map_y = map_landmarks.landmark_list[map_id - 1].y_f;
                obs_2_map_distance.push_back(dist(obs_2_map_x, obs_2_map_y, map_x, map_y));
            }
            // Find the minimum element in the distance
            auto min_loc = min_element(obs_2_map_distance.begin(), obs_2_map_distance.end());
            //find the index of minimum element and that is the nearest landmark for that observation
            obs_2_map_nn.push_back(landmark_2_particle_sensed[distance(obs_2_map_distance.begin(), min_loc)]);
            //cout << "The nearest neighbour for " << observations[j].x << " is " << landmark_2_particle_sensed[distance(obs_2_map_distance.begin(), min_loc)] << endl;

            // Now every transformed observation has a nearest neighbour landmark associated with it.
            // Let's calculate weight
            // Assuming std for x and y are constants
            double mul_constant = 1 / (2 * M_PI * std_landmark[0] * std_landmark[1]);
            double x_den = 1/(2 * std_landmark[0] * std_landmark[0]);
            double y_den = 1/(2 * std_landmark[1] * std_landmark[1]);

            double x_diff = obs_2_map_x - map_landmarks.landmark_list[obs_2_map_nn[j] -1].x_f;
            double y_diff = obs_2_map_y - map_landmarks.landmark_list[obs_2_map_nn[j] -1].y_f;

            multiVarGD *= mul_constant * exp(-1 * ( (x_diff * x_diff)/x_den + (y_diff * y_diff)/y_den) );
        }
    particles[i].weight = multiVarGD;
    weights[i] = multiVarGD;
    //cout << "Particle : " << i << " - " << particles[i].associations.size() << endl;
    }
    cout << " ParticleFilter::updateWeights - call over \n";
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
    cout << "ParticleFilter::resample called" <<endl;

    //Create a vector to hold temporary particles
    vector<Particle> temp_particles(num_particles);
    discrete_distribution<> d(weights.begin(), weights.end());

    for (int i = 0; i < num_particles; i++){
        temp_particles[i] = particles[d(gen)];
    }
    particles = temp_particles;
    cout << "ParticleFilter::resample - call over \n";
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	cout << "ParticleFilter::SetAssociations called" <<endl;
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;
    cout << "ParticleFilter::SetAssociations - call over" <<endl;
 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
