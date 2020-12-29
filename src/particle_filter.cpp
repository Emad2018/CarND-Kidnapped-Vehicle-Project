/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"
#include <random> // Need this for sampling from distributions

using std::normal_distribution;
using std::string;
using std::vector;

void ParticleFilter::init(double x, double y, double theta, double std[])
{
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */

  num_particles = 500;            //Set the number of particles
  std::default_random_engine gen; //   where "gen" is the random engine initialized earlier.
  double std_x, std_y, std_theta; // Standard deviations for x, y, and theta

  // Set standard deviations for x, y, and theta
  std_x = std[0];
  std_y = std[1];
  std_theta = std[2];

  // This line creates a normal (Gaussian) distribution for x
  normal_distribution<double> dist_x(x, std_x);

  //create normal distributions for y and theta
  normal_distribution<double> dist_y(y, std_y);
  normal_distribution<double> dist_theta(theta, std_theta);

  for (int i = 0; i < num_particles; ++i)
  {
    double sample_x, sample_y, sample_theta;

    //Sample from these normal distributions like this:
    //   sample_x = dist_x(gen);
    Particle particle;

    sample_x = dist_x(gen);
    sample_y = dist_y(gen);
    sample_theta = dist_theta(gen);
    particle.id = i;
    particle.weight = 1;
    particle.x = sample_x;
    particle.y = sample_y;
    particle.theta = sample_theta;
    particles.push_back(particle);
  }
  is_initialized = true;
  //print();
}

void ParticleFilter::prediction(double delta_t, double std_pos[],
                                double velocity, double yaw_rate)
{
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  std::default_random_engine gen;
  double std_x, std_y, std_theta;
  std_x = std_pos[0];
  std_y = std_pos[1];
  std_theta = std_pos[2];
  normal_distribution<double> dist_x(0, std_x);

  //create normal distributions for y and theta
  normal_distribution<double> dist_y(0, std_y);
  normal_distribution<double> dist_theta(0, std_theta);
  for (int i = 0; i < num_particles; ++i)
  {
    double dv = velocity / yaw_rate;
    double dtheta = particles[i].theta + yaw_rate * delta_t;
    particles[i].x = particles[i].x + (dv * (sin(dtheta) - sin(particles[i].theta))) + dist_x(gen);
    particles[i].y = particles[i].y + (dv * (cos(particles[i].theta) - cos(dtheta))) + dist_y(gen);
    particles[i].theta = dtheta + dist_theta(gen);
  }
  //print();
}

std::vector<std::vector<LandmarkObs>> ParticleFilter::dataAssociation(vector<LandmarkObs> predicted,
                                                                      vector<LandmarkObs> &observations)
{
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */

  vector<vector<LandmarkObs>> matches;
  //std::cout << "predicted.size= " << predicted.size() << "\n";
  //std::cout << "observations.size= " << observations.size() << "\n";
  for (int i = 0; i < observations.size(); i++)
  {
    float ldist;
    float small_dist = 10000;
    int related_index;
    bool found = false;
    vector<LandmarkObs> match;
    for (int j = 0; j < predicted.size(); ++j)
    {
      ldist = dist(predicted[i].x, predicted[i].y, observations[j].x, observations[j].y);

      if (ldist < small_dist)
      {
        small_dist = ldist;
        related_index = j;
        found = true;
        //std::cout << "ldist= " << ldist << "\n";
      }
    }
    if (found)
    {
      match.push_back(predicted[i]);
      match.push_back(observations[related_index]);
      matches.push_back(match);
    }
  }
  return matches;
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const vector<LandmarkObs> &observations,
                                   const Map &map_landmarks)
{
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */

  for (int i = 0; i < num_particles; ++i)
  {
    float x = particles[i].x;
    float y = particles[i].y;
    float theta = particles[i].theta;
    float prob = 1;
    vector<LandmarkObs> predicted;
    vector<LandmarkObs> observations_transformed;
    for (Map::single_landmark_s current_landmark : map_landmarks.landmark_list)
    {
      if ((fabs((x - current_landmark.x_f)) <= sensor_range) && (fabs((y - current_landmark.y_f)) <= sensor_range))
      {
        LandmarkObs closeest_landmark;
        closeest_landmark.id = current_landmark.id_i;
        closeest_landmark.x = current_landmark.x_f;
        closeest_landmark.y = current_landmark.y_f;
        predicted.push_back(closeest_landmark);
      }
    }

    for (LandmarkObs current_observation : observations)
    {
      LandmarkObs observation_transformed;
      //transform observation to map coordenates

      float xm = x + (current_observation.x * cos(theta)) - (current_observation.y * sin(theta));
      float ym = y + (current_observation.y * cos(theta)) + (current_observation.x * sin(theta));
      observation_transformed.id = current_observation.id;
      observation_transformed.x = xm;
      observation_transformed.y = ym;
      observations_transformed.push_back(observation_transformed);
    }
    vector<vector<LandmarkObs>> matches = dataAssociation(predicted, observations_transformed);

    if (matches.size() > 0)
    {
      for (int i = 0; i < matches.size(); i++)
      {
        LandmarkObs match_predection = matches[i][0];
        LandmarkObs match_observation = matches[i][1];
        float xdiff = abs(x - match_predection.x);
        float ydiff = abs(y - match_predection.y);
        prob *= Gaussian(xdiff, ydiff, match_observation.x, match_observation.y, std_landmark);
      }
    }
    else
    {

      prob = 0;
    }
    particles[i].weight = prob;
    //get prob for every partical
    if (prob > 0)
    {
      std::cout << "  predicted x= " << matches[0][0].x << "  predicted y= " << matches[0][0].y << "\n";
      std::cout << "  observations x= " << matches[0][1].x << "  observations y= " << matches[0][1].y << "\n";
      std::cout << "matches.size()= " << matches.size() << "\n";
      std::cout << prob << "\n";
    }
  }
  /*
   if ((fabs((particle_x - current_landmark.x_f)) <= sensor_range) && (fabs((particle_y - current_landmark.y_f)) <= sensor_range))
  */
  //print();
}

void ParticleFilter::resample()
{
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  /*
  p3 = []
index=int(random.random()*len(w))
beta=0
max_double_W=2*max(w)
for i in range(len(w)):
    beta +=random.random()*max_double_W
    while w[index%N] < beta:
        beta = beta - w[index%N]
        index = index + 1
    p3.append(p[index%N])
p = p3
*/
  std::default_random_engine gen;
  normal_distribution<double> dist_x(0, 1);

  std::vector<Particle> resample_particles;
  int N = particles.size();
  int index = abs((int)dist_x(gen) * N);
  double max_W = 0;
  double beta = 0;
  for (Particle particle : particles)
  {
    if (particle.weight > max_W)
    {
      max_W = particle.weight;
    }
  }

  for (int i = 0; i < N; ++i)
  {
    beta += abs(dist_x(gen)) * 2 * max_W;
    while (particles[index % N].weight < beta)
    {
      beta = beta - particles[index % N].weight;
      index++;
    }
    resample_particles.push_back(particles[index % N]);
  }
  particles = resample_particles;
}

void ParticleFilter::SetAssociations(Particle &particle,
                                     const vector<int> &associations,
                                     const vector<double> &sense_x,
                                     const vector<double> &sense_y)
{
  // particle: the particle to which assign each listed association,
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations = associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1); // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord)
{
  vector<double> v;

  if (coord == "X")
  {
    v = best.sense_x;
  }
  else
  {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1); // get rid of the trailing space
  return s;
}

void ParticleFilter::print()
{
  for (Particle particle : particles)
  {

    printf("id= %d , x= %f , y=%f , theta= %f , w= %f \n", particle.id, particle.x, particle.y, particle.theta, particle.weight);
  }
}