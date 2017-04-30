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

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// Set the number of particles. Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1.
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	// Initialization
	//   Set number of particles
	num_particles = 100;

	//   Resize weights matrix according to number of particles
	weights.resize(num_particles);

	//   Create Gaussian normal distributions for x, y, and theta
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);

	//   Create a random generator object
	default_random_engine gen;

	//   Generate random Particles with the Gaussian distributions
	for (int i = 0; i < num_particles, i++) {

		Particle p;

		// Sample particle with random values and intial weight of 1
		p.x = dist_x(gen);
		p.y = dist_y(gen);
		p.theta = dist_theta(gen);
		p.weight = 1;

		// Append to particles vector
		particles.push_back(p);

	}

	is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	//   Create Gaussian normal distributions for noise in x, y, and theta
	normal_distribution<double> noise_x(0, std_pos[0]);
	normal_distribution<double> noise_y(0, std_pos[1]);
	normal_distribution<double> noise_theta(0, std_pos[2]);

	//   Create a random generator object
	default_random_engine gen;

	//   If yaw_rate is zero, set it to a non-zero value
	if(abs(yaw_rate) < 0.0001) {
		yaw_rate = 0.0001;
	} else {
		// Generate sample Particles with added noise in x, y, and theta (Bicycle Model)
		for (int i = 0; i < len(particles); i++) {

			// Update x for non-zero yaw_rate
			p.x = p.x + (velocity / yaw_rate) * (sin(p.theta + (yaw_rate * delta_t)) - sin(p.theta))
			// Add noise
			p.x += noise_x(gen);

			// Update y for non-zero yaw_rate
			p.y = p.y + (velocity / yaw_rate) * (cos(p.theta + (yaw_rate * delta_t)) - cos(p.theta))
			// Add noise
			p.y += noise_y(gen);

			// Update theta for non-zero yaw_rate
			p.theta = p.theta + (yaw_rate * delta_t);
			// Add noise
			p.theta = noise_theta(gen);
		}
	}

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
	//   implement this method and use it as a helper during the updateWeights phase.
	for (int i = 0; i < observations.size(); i++) {

		double dist = 0;
		int nearest_landmark = -1;

		// Iterate through landmarks to find nearest
		for (int j = 0; j < predicted.size(); j++) {

			// Calculate Euclidian distance
			double dist_eucl = dist(observations[i].x, observations[i].y, predicted[i].x, predicted[i].y);

			// Handle special case
			if (dist_eucl < dist) {
				dist = dist_eucl;
				nearest_landmark = j;
			}
		}

		// Assign nearest landmark id to observation id
		observations[i].id = predicted[nearest_landmark].id;

	}

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation
	//   3.33. Note that you'll need to switch the minus sign in that equation to a plus to account
	//   for the fact that the map's y-axis actually points downwards.)
	//   http://planning.cs.uiuc.edu/node99.html

	// Clear weights from previous steps
	weights.clear();

	// Iterate through all particles
	for (int i = 0; i < particles.size(); i++) {

		// Transform observation coordinates from vehicle coordinates to map coordinates
		std::vector<LandmarkObs> obs_map;
		for (int j = 0; j < observations.size(); j++) {
			if (dist(observations[j].x, observations[j].y, 0, 0) <= sensor_range) {
				LandmarkObs obs_temp;
				obs_temp.x = particles[i].x + observations[j].x * cos(particles[i].theta) - observations[j].y * sin(particles[i].theta);
				obs_temp.y = particles[i].y + observations[j].x * sin(particles[i].theta) - observations[j].y * cos(particles[i].theta);
				obs_temp.id = -1;
				obs_map.push_back(obs_temp);
			}
		}

		// Create a list of nearest landmarks in map coordinates
		std::vector<LandmarkObs> nearest_landmarks;
		for (int j = 0; j < map_landmarks.landmark_list.size(); j++) {
			if (dist(particles[i].x, particles[i].y, map_landmarks.landmark_list[j].x_f, map_landmarks.landmark_list[j].y_f) <= sensor_range) {
				LandmarkObs obs_temp;
				obs_temp.x = map_landmarks.landmark_list[j].x_f;
				obs_temp.y = map_landmarks.landmark_list[j].y_f;
				obs_temp.id = map_landmarks.landmark_list[j].id_i;
				nearest_landmarks.push_back(obs_temp);
			}
		}

		// Find the nearest landmark id for each observation
		dataAssociation(nearest_landmarks, obs_map);

		// Calculate and assign weights by multiplying bivariate probability of each observation
		double weight = 1;
		for (int j = 0; j < nearest_landmarks.size(); j++) {
			double dist_min = 0.00001;
			int min_k = -1;

			for (int k = 0; k < obs_map_coord.size(); k++) {
				// Find minimum distance for observations that are nearest to each landmark
				if(obs_map[k].id == nearest_landmarks[j].id) {
					double dist_eval = dist(nearest_landmarks[j].x, nearest_landmarks[j].y, obs_map[k].x, obs_map[k].y);
					if (dist_eval < dist_min) {
						dist_min = dist_eval;
						min_k = k;
					}
				}
			}
			if (min_k != -1) {
				// Define variables for readability
				double x_mu_x_sqd = (obs_map[min_k].x - nearest_landmarks[j].x) * (obs_map[min_k].x - nearest_landmarks[j].x)
				double y_mu_y_sqd = (obs_map[min_k].y - nearest_landmarks[j].y) * (obs_map[min_k].y - nearest_landmarks[j].y)
				double x_sig_sqd = std_landmark[0] * std_landmark[0];
				double y_sig_sqd = std_landmark[1] * std_landmark[1];

				double exp_term = exp(-(x_mu_x_sqd)/(2 * x_sig_sqd)) + ((y_mu_y_sqd)/(2 * y_sig_sqd)));
				weight = weight * (1/(2 * M_PI * std_landmark[0] * std_landmark[1])) * exp_term;
			}
		}

		// Update weights
		weights.push_back(weight);
		particles[i].weight = weight;

	}

}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

}

void ParticleFilter::write(std::string filename) {
	// You don't need to modify this file.
	std::ofstream dataFile;
	dataFile.open(filename, std::ios::app);
	for (int i = 0; i < num_particles; ++i) {
		dataFile << particles[i].x << " " << particles[i].y << " " << particles[i].theta << "\n";
	}
	dataFile.close();
}
