/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 * 	 Edited on: Apr 30, 2017
 *      Editor: Neil Hiddink
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

  // Set the number of particles
  num_particles = 20;

  // Create Gaussian noise distributions
  normal_distribution<double> dist_x(0, std[0]);
  normal_distribution<double> dist_y(0, std[1]);
  normal_distribution<double> dist_theta(0, std[2]);

	// Create random number generator
	default_random_engine gen;

  // Set particles iniital particle position and noise, each with weight of 1
  for (int i = 0; i < num_particles; ++i) {

		Particle p;
    p.id = i;
    p.x = x + dist_x(gen);
    p.y = y + dist_y(gen);
    p.theta = theta + dist_theta(gen);
    p.weight = 1;

		weights.push_back(p.weight);
    particles.push_back(p);
  }

	// The particle filter is now initialized
  is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

  // Create Gaussian noise distributions
  normal_distribution<double> noise_x(0, std_pos[0]);
  normal_distribution<double> noise_y(0, std_pos[1]);
  normal_distribution<double> noise_theta(0, std_pos[2]);

	// Create random number generator
  default_random_engine gen;

  for (unsigned int i=0; i < particles.size(); ++i) {

    // Define temporary variables for predictions
    double x_pred;
    double y_pred;
    double theta_pred;

    // Update the state based on value of yaw rate
    if (yaw_rate < 0.001) {
      x_pred = particles[i].x + velocity * cos(particles[i].theta) * delta_t;
      y_pred = particles[i].y + velocity * sin(particles[i].theta) * delta_t;
      theta_pred = particles[i].theta + yaw_rate * delta_t;
    } else {
      x_pred = particles[i].x + (velocity / yaw_rate) * (sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta));
      y_pred = particles[i].y + (velocity / yaw_rate) * (-cos(particles[i].theta + yaw_rate * delta_t) + cos(particles[i].theta));
      theta_pred = particles[i].theta + yaw_rate * delta_t;
    }

    // Update particle with noise added
    particles[i].x = x_pred + noise_x(gen);
    particles[i].y = y_pred + noise_y(gen);
    particles[i].theta = theta_pred + noise_theta(gen);

  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
	//   implement this method and use it as a helper during the updateWeights phase.

  // Iterate through observations
  for (unsigned i = 0; i < observations.size(); ++i) {

    // Define temporary variables for finding predicted measurements
    double dist_current = 1e6;
    int nearest_landmark = -1;

    // Iterate through predicted measurements to find nearest landmarks
    for (unsigned j = 0; j < predicted.size(); ++j) {

      // Calculate Euclidian distance between observations and predictions
      double dist_eucl = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);

      // assign min
      if (dist_eucl < dist_current) {
        dist_current = dist_eucl;
        nearest_landmark = j;
      }
    }
    // assign the closest id to the obeservation
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


  // Empty the list of weights for all particles
  weights.clear();

  // Iterate through all particles
  for (unsigned i = 0; i < particles.size(); ++i) {

    // Transform observation coordinates from vehicle to map
    std::vector<LandmarkObs> obs_map;
    for (unsigned int j = 0; j < observations.size(); ++j) {
      if (dist(observations[j].x, observations[j].y, 0, 0) <= sensor_range) {
        LandmarkObs obs;
        obs.x = particles[i].x + observations[j].x * cos(particles[i].theta) - observations[j].y * sin(particles[i].theta);
        obs.y = particles[i].y + observations[j].x * sin(particles[i].theta) + observations[j].y * cos(particles[i].theta);
        obs.id = -1;
        obs_map.push_back(obs);
      }
    }

    // Create a list of nearest landmarks in map coordinates
    std::vector<LandmarkObs> nearest_landmarks;
    for (unsigned int j = 0; j < map_landmarks.landmark_list.size(); ++j) {
      if (dist(particles[i].x, particles[i].y, map_landmarks.landmark_list[j].x_f, map_landmarks.landmark_list[j].y_f) <= sensor_range) {
        LandmarkObs obs;
        obs.x = map_landmarks.landmark_list[j].x_f;
        obs.y = map_landmarks.landmark_list[j].y_f;
        obs.id = map_landmarks.landmark_list[j].id_i;
        nearest_landmarks.push_back(obs);
      }
    }

    // Find the nearest landmark id for each observaton
    dataAssociation(nearest_landmarks, obs_map);

    // Calculate weights by factoring in multivariate Gaussian probabilities
    double weight = 1;
    for (unsigned int j = 0; j < nearest_landmarks.size(); j++) {
      double dist_min = 1e6;
      int min_k = -1;

			// Iterate through map coordinates
      for (unsigned int k = 0; k < obs_map.size(); ++k) {
        // Iterate through nearest landmark observations to find minimum distance
        if (obs_map[k].id == nearest_landmarks[j].id) {
          double dist_eucl = dist(nearest_landmarks[j].x, nearest_landmarks[j].y, obs_map[k].x, obs_map[k].y);
          if (dist_eucl < dist_min) {
            dist_min = dist_eucl;
            min_k = k;
          }
        }
      }
      if (min_k != -1) {
        weight *= multi_gauss_prob(obs_map[min_k].x, obs_map[min_k].y, nearest_landmarks[j].x, nearest_landmarks[j].y, std_landmark[0], std_landmark[1]);
      }
    }

    // Update particles with correct weights
    weights.push_back(weight);
    particles[i].weight = weight;

  }

}

void ParticleFilter::resample() {
	// Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

  // Create random number generator
  default_random_engine gen;

  auto first = weights.cbegin();
  auto last = weights.cend();
  auto count = distance(first, last);

	// Set up discrete distribution using reference link in note above
	discrete_distribution<int> dist(
    count,
    -0.5,
    -0.5 + count,
    [&first](size_t i)
  {
    return *std::next(first, i);
  });

  // Resample particles
  std::vector<Particle> resampled_particles;
  for (int i = 0; i<num_particles; ++i) {
    resampled_particles.push_back(particles[dist(gen)]);
  }
  particles = resampled_particles;

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
