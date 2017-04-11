#ifndef TEMPLATES_H_
#define TEMPLATES_H_

#define _USE_MATH_DEFINES
#include <cmath>

// Convert radians into [-pi; pi] range.
template<typename T>
T normalizeRadiansPiToMinusPi(T rad) {
  static const T PI2 = 2.*M_PI;
  // Copy the sign of the value in radians to the value of pi.
  T signed_pi = std::copysign(M_PI,rad);
  // Set the value of difference to the appropriate signed value between pi and -pi.
  rad = std::fmod(rad + signed_pi,(PI2)) - signed_pi;
  return rad;
}


#endif /* TEMPLATES_H_ */
